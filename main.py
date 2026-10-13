import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

"""Entry point: run the conversational swarm (N agents x T rounds) end-to-end on the fixed labeled split."""

import argparse
import json
import shutil

from agents.prompts import PROMPTS_PATH, PROMPTS_SHA
from agents.swarm_agent import make_agent_llm, make_llm
from config import AgentGroup, Settings
from corpus.ingest import build_index
from data.dataset import build_data_cache
from orchestration.graph import (
    build_graph, create_initial_state, new_run_id, open_checkpointer, recursion_limit,
)

RUN_CONFIG = "run_config.json"  # what a resumed run must keep: every agent's model, and the target rounds


def _check_models(settings: Settings) -> None:
    """Fail now, not mid-round: every model needs an API key (except local providers) and its
    LangChain provider package installed."""
    missing = sorted({
        g.llm_provider for g in settings.agents if not g.api_key and g.llm_provider not in ("ollama",)
    })
    if missing:
        raise SystemExit(
            "No API key for agent provider(s) " + ", ".join(missing) + ". Add "
            + ", ".join(f"{m.upper()}_API_KEY" for m in missing) + " to .env (or set api_key in config.agents)."
        )
    try:
        seen: set[str] = set()
        for i, g in enumerate(settings.agent_models()):
            if g.label not in seen:
                seen.add(g.label)
                make_agent_llm(settings, i)
        make_llm(settings, 2)
        make_llm(settings, 3)
    except Exception as exc:  # noqa: BLE001 - e.g. a provider package that is not installed
        raise SystemExit(f"Could not create a chat model: {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoLSM-Swarm conversational swarm")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="continue an existing run (under <runs_dir>/<RUN_ID>) from its last checkpoint, "
        "mid-conversation included; with --rounds, also extend it",
    )
    parser.add_argument(
        "--agents", type=int, default=None,
        help="run N agents, all on the model of the first group in config.agents (default: the groups as "
        "configured; ignored with --resume)",
    )
    parser.add_argument("--rounds", type=int, default=None, help="number of rounds (default from config; with --resume: the new target)")
    parser.add_argument("--cycles", type=int, default=None, help="conversation cycles per round (default from config)")
    parser.add_argument("--max-train-tiles", type=int, default=None, help="ignored with --resume -- data is reused from the original run")
    parser.add_argument("--n-seeds", type=int, default=None, help="evaluate each pipeline under up to N of the seeds registered in config.eval_seeds (default: all of them)")
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=None,
        help="stop early after this many consecutive rounds with no global-best Dice improvement (default from config)",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="force rebuild of the RAG corpus index")
    parser.add_argument("--show-graph", action="store_true", help="print the swarm graph as mermaid and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    settings = Settings()

    if args.agents is not None and args.resume is None:
        settings.agents = [settings.agents[0].model_copy(update={"n_agents": args.agents})]
    if args.show_graph:
        print(build_graph(settings, settings.n_agents).get_graph().draw_mermaid())
        return

    if args.rounds is not None:
        settings.n_rounds = args.rounds
    if args.cycles is not None:
        settings.conversation_cycles = args.cycles
    if args.max_train_tiles is not None:
        settings.max_train_tiles = args.max_train_tiles
    if args.plateau_patience is not None:
        settings.plateau_patience = args.plateau_patience
    if args.n_seeds is not None:
        if not 1 <= args.n_seeds <= len(settings.eval_seeds):
            raise SystemExit(f"--n-seeds must be between 1 and {len(settings.eval_seeds)} (the seeds registered in config.eval_seeds)")
        settings.n_seeds = args.n_seeds

    if settings.use_rag:
        index_marker = settings.vector_store_dir / "chroma.sqlite3"
        if args.rebuild_index or not index_marker.exists():
            print(f"Building RAG index from {settings.corpus_json_dir} ...")
            n_docs = build_index(settings)
            print(f"Indexed {n_docs} papers.")

    if args.resume is not None:
        run_id = args.resume
        run_dir = settings.runs_dir / run_id
        if not (run_dir / "checkpoints.sqlite").exists():
            raise SystemExit(f"No checkpoints.sqlite for run {run_id!r} under {run_dir}.")
        run_config = json.loads((run_dir / RUN_CONFIG).read_text(encoding="utf-8"))
        if args.agents is not None:
            print(f"--agents {args.agents} ignored: the agent count comes from the resumed run.")
        if "agents" in run_config:  # every agent keeps the model it started with
            settings.agents = [AgentGroup(n_agents=1, **a) for a in run_config["agents"]]
            settings.fill_agent_api_keys()
        elif run_config["n_agents"] != settings.n_agents:
            raise SystemExit(f"Run {run_id!r} has {run_config['n_agents']} agents; config.agents has {settings.n_agents}.")
        _check_models(settings)
        if args.rounds is None:
            settings.n_rounds = run_config["n_rounds"]
        run_config["n_rounds"] = settings.n_rounds
        (run_dir / RUN_CONFIG).write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        graph = build_graph(settings, settings.n_agents, open_checkpointer(run_dir))
        config = {"configurable": {"thread_id": run_id}}
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise SystemExit(f"Run {run_id!r} has no checkpoint to resume from.")
        done_round = snapshot.values["round"]
        if not snapshot.next:  # the run already finalized
            if settings.n_rounds <= done_round + 1:
                raise SystemExit(
                    f"Run {run_id!r} already finished after round {done_round}. Pass --rounds greater than "
                    f"{done_round + 1} to extend it."
                )
            # Re-enter right after the last observer: the conversation of the last round runs, then
            # the new rounds. A fresh plateau window -- the user is explicitly asking for more rounds.
            graph.update_state(config, {"rounds_without_improvement": 0, "stop_reason": None}, as_node="observer")
            print(f"Extending finished run {run_id}: round {done_round} -> {settings.n_rounds - 1}")
        else:
            print(f"Resuming run {run_id} at {list(snapshot.next)} (round {done_round}, target {settings.n_rounds} rounds)")
        used = run_dir / "prompts_used.yaml"
        if used.exists() and used.read_bytes() != PROMPTS_PATH.read_bytes():
            print(
                f"WARNING: prompts file {PROMPTS_PATH.name} differs from the {used.name} stored with run "
                f"{run_id}; everything from here on uses the current text."
            )
        state_in = None
        rounds_left = settings.n_rounds - done_round
    else:
        run_id = new_run_id()
        run_dir = settings.runs_dir / run_id
        _check_models(settings)
        print(
            f"Run {run_id}: {settings.n_agents} agent(s) x {settings.n_rounds} round(s), "
            f"{settings.conversation_cycles} conversation cycle(s) per round"
        )
        for g in settings.agents:
            print(f"  {g.n_agents} x {g.label}")
        data_npz_path = run_dir / "_data_cache.npz"
        build_data_cache(settings.dataset_name, settings.dataset_dir, data_npz_path, settings.max_train_tiles)
        shutil.copy2(PROMPTS_PATH, run_dir / "prompts_used.yaml")
        (run_dir / RUN_CONFIG).write_text(
            json.dumps({
                "n_agents": settings.n_agents,
                "n_rounds": settings.n_rounds,
                "max_train_tiles": settings.max_train_tiles,  # to rebuild the same training set offline
                "eval_seeds": settings.active_seeds(),
                "agents": [  # one entry per agent_idx; API keys are never written
                    g.model_dump(include={"llm_provider", "model_name", "max_tokens", "model_kwargs"}) for g in settings.agent_models()
                ],
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Prompts: {PROMPTS_PATH.name} (sha {PROMPTS_SHA})")

        graph = build_graph(settings, settings.n_agents, open_checkpointer(run_dir))
        config = {"configurable": {"thread_id": run_id}}
        state_in = create_initial_state(run_id, [g.label for g in settings.agent_models()], data_npz_path)
        rounds_left = settings.n_rounds

    config["recursion_limit"] = recursion_limit(settings, settings.n_agents, rounds_left)
    final_state = graph.invoke(state_in, config=config)

    print(json.dumps({"g_best_score": final_state["g_best_score"], "best_agent": final_state["g_best_agent"]}, indent=2))
    print(f"Ledger: {run_dir / 'ledger.jsonl'}")


if __name__ == "__main__":
    main()
