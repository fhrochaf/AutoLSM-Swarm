import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

"""Entry point: run N agents x T rounds end-to-end on the fixed labeled split."""

import argparse
import json
import sys

from config import Settings
from corpus.ingest import build_index
from data.dataset import build_data_cache
from orchestration.graph import build_graph, create_initial_state, load_run_state, new_run_id


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoLSM-Swarm round loop")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="keep training an existing run (under runs/<RUN_ID>) instead of starting a new one",
    )
    parser.add_argument("--agents", type=int, default=None, help="number of agents (default from config; ignored with --resume)")
    parser.add_argument("--rounds", type=int, default=None, help="number of rounds (default from config; required with --resume -- the target round to train up to)")
    parser.add_argument("--max-train-tiles", type=int, default=None, help="ignored with --resume -- data is reused from the original run")
    parser.add_argument("--max-val-tiles", type=int, default=None, help="ignored with --resume -- data is reused from the original run")
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=None,
        help="stop early after this many consecutive rounds with no global-best Dice improvement (default from config)",
    )
    parser.add_argument(
        "--rebuild-index", action="store_true", help="force rebuild of the RAG corpus index"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    settings = Settings()

    if args.resume is not None:
        if args.agents is not None:
            print(f"--agents {args.agents} ignored: agent count comes from the resumed run.")
        if args.max_train_tiles is not None:
            print(f"--max-train-tiles {args.max_train_tiles} ignored: data is reused from the resumed run.")
        if args.max_val_tiles is not None:
            print(f"--max-val-tiles {args.max_val_tiles} ignored: data is reused from the resumed run.")
        if args.rounds is None:
            raise SystemExit(
                "--rounds is required with --resume: the target round count to keep "
                "training up to (e.g. --resume <run_id> --rounds 10)."
            )
    if args.rounds is not None:
        settings.n_rounds = args.rounds
    if args.max_train_tiles is not None:
        settings.max_train_tiles = args.max_train_tiles
    if args.max_val_tiles is not None:
        settings.max_val_tiles = args.max_val_tiles
    if args.plateau_patience is not None:
        settings.plateau_patience = args.plateau_patience
    if args.agents is not None and args.resume is None:
        settings.n_agents = args.agents

    if not settings.api_key:
        raise SystemExit(
            f"No API key found for llm_provider={settings.llm_provider!r}. Add "
            f"{settings.llm_provider.upper()}_API_KEY (or API_KEY) to .env before running the swarm."
        )

    index_marker = settings.vector_store_dir / "chroma.sqlite3"
    if args.rebuild_index or not index_marker.exists():
        print(f"Building RAG index from {settings.corpus_json_dir} ...")
        n_docs = build_index(settings)
        print(f"Indexed {n_docs} papers.")

    if args.resume is not None:
        run_id = args.resume
        state = load_run_state(run_id, settings, settings.n_rounds)
        mid_round = f", {len(state.in_progress_agents)} agent(s) already done this round" if state.in_progress_agents else ""
        print(
            f"Resuming run {run_id}: {len(state.agents)} agent(s), "
            f"round {state.round} -> {settings.n_rounds}{mid_round}"
        )
        recursion_limit = max(settings.n_rounds - state.round, 1) + 10
    else:
        run_id = new_run_id()
        print(f"Run {run_id}: {settings.n_agents} agent(s) x {settings.n_rounds} round(s)")

        data_npz_path = settings.runs_dir / run_id / "_data_cache.npz"

        # Loading dataset to cache
        build_data_cache(
            settings.dataset_name,
            settings.dataset_dir,
            data_npz_path,
            settings.max_train_tiles,
            settings.max_val_tiles,
        )

        state = create_initial_state(settings, run_id, data_npz_path)
        recursion_limit = settings.n_rounds + 10

    graph = build_graph(settings)
    final_state = graph.invoke(state, config={"recursion_limit": recursion_limit})

    print(json.dumps({"g_best_score": final_state["g_best_score"]}, indent=2))
    print(f"Ledger: {settings.runs_dir / run_id / 'ledger.jsonl'}")


if __name__ == "__main__":
    main()
