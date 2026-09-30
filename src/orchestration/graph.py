"""The round loop, wired as a LangGraph StateGraph.

Round 0 seeds every agent from the literature corpus and generates its first
pipeline.py. Later rounds first run one peer review (every agent's behavior on the same
validation tiles), then ask peer_review to revise each agent's pipeline.py against that
review, the swarm's global-best and its own personal-best. pipeline.py is the only
artifact an agent owns; there is no separate strategy document.

SwarmState/AgentState are pydantic models (orchestration/state.py). LangGraph passes an
actual model instance into each node (attribute access below), and node functions
return a plain dict of the fields that changed -- LangGraph merges that back into the
persisted state. `graph.invoke(...)` itself always returns a plain dict regardless of
the schema type, not a model instance.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from langchain.chat_models import init_chat_model
import numpy as np
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from tqdm import tqdm

from agents import artifacts
from agents.codegen import generate_initial_pipeline
from agents.prompts import RETRIEVAL_QUERY
from agents.runner import run_agent_round
from config import Settings
from corpus.retrieve import format_for_prompt, retrieve_diverse
from eval import observe
from orchestration import peer_review, registry
from orchestration.state import AgentState, SwarmState
from utils import git_version


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _append_ledger(runs_dir: Path, run_id: str, entry: dict) -> None:
    ledger_path = runs_dir / run_id / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _read_ledger(runs_dir: Path, run_id: str) -> list[dict]:
    ledger_path = runs_dir / run_id / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    with ledger_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_initial_state(settings: Settings, run_id: str, data_npz_path: Path) -> SwarmState:
    agents = [AgentState(agent_idx=i) for i in range(settings.n_agents)]
    return SwarmState(
        run_id=run_id,
        n_rounds=settings.n_rounds,
        data_npz_path=str(data_npz_path),
        agents=agents,
    )


def load_run_state(run_id: str, settings: Settings, target_n_rounds: int) -> SwarmState:
    """Reconstruct a SwarmState from an existing run's checkpoint to keep training it.

    Reads only runs/<run_id>/state_snapshot.json -- the single, always-overwritten
    checkpoint _run_round_node writes after every agent (not just every round) -- and
    the run's kept dataset cache. No earlier round's files are read: the snapshot
    already holds every agent's current velocity/p_best/pipeline code and the swarm's
    global-best, so there is nothing to replay. If the snapshot was taken mid-round
    (some but not all agents done), `state.in_progress_agents` carries those agents'
    already-finished results and `_run_round_node` skips re-running them.
    """
    run_dir = settings.runs_dir / run_id
    snapshot_path = run_dir / "state_snapshot.json"
    if not snapshot_path.exists():
        raise SystemExit(
            f"No state_snapshot.json found for run {run_id!r} under {run_dir}. "
            "Either the run id is wrong, or it predates the resume feature."
        )

    data_npz_path = run_dir / "_data_cache.npz"
    if not data_npz_path.exists():
        raise SystemExit(
            f"No cached dataset (_data_cache.npz) found for run {run_id!r} under "
            f"{run_dir}. Resuming requires the exact train/val tiles the original "
            "rounds were scored against; it cannot be rebuilt from scratch."
        )

    state = SwarmState.model_validate_json(snapshot_path.read_text(encoding="utf-8"))

    if target_n_rounds <= state.round:
        raise SystemExit(
            f"Run {run_id!r} already completed round {state.round - 1} "
            f"(next round would be {state.round}). Pass --rounds greater than "
            f"{state.round} to continue training it."
        )

    return state.model_copy(
        update={
            "n_rounds": target_n_rounds,
            "data_npz_path": str(data_npz_path),
            # A prior plateau stop shouldn't immediately re-trigger on resume -- the
            # user is explicitly asking for more rounds, so give it a fresh patience
            # window. The plateau rule itself (settings.plateau_patience) is unchanged.
            "rounds_without_improvement": 0,
        }
    )


def _make_llm(provider: str, model: str, api_key: str, max_tokens: int):
    return init_chat_model(
        model=model,
        model_provider=provider,
        api_key=api_key or None,
        # temperature=settings.temperature,
        max_tokens=max_tokens,
    )


# Compact per-run behavior figures stored in the ledger (the full tables live in each
# round's peer_review_tables.md), so a run can be analyzed without reloading predictions.
_LEDGER_METRICS = (
    "precision", "recall", "tile_precision", "tile_recall", "fp_tiles", "object_recall",
    "object_precision", "pred_pos_rate", "true_pos_rate",
)


def _run_round_node(state: SwarmState, settings: Settings) -> dict:
    # Primary: Reflect / GroundReflection / VelocityUpdate. Peer review (role 2): the
    # per-round comparative report. Code (role 3): every pipeline.py write and fix.
    llm = _make_llm(settings.llm_provider, settings.model_name, settings.api_key, settings.max_tokens)
    peer_review_llm = _make_llm(settings.llm_provider_2, settings.model_name_2, settings.api_key_2, settings.max_tokens_2)
    code_llm = _make_llm(settings.llm_provider_3, settings.model_name_3, settings.api_key_3, settings.max_tokens_3)

    round_idx = state.round
    data_npz_path = Path(state.data_npz_path)
    snapshot_path = settings.runs_dir / state.run_id / "state_snapshot.json"
    y_val = np.load(data_npz_path)["y_val"]
    size_edges = observe.size_edges(y_val)
    seeds = settings.active_seeds()

    # Agents already checkpointed for this round (e.g. resuming after a mid-round
    # interrupt) are skipped -- their results are reused as-is instead of rerun.
    in_progress = dict(state.in_progress_agents)
    remaining = [a for a in state.agents if a.agent_idx not in in_progress]

    def _checkpoint() -> None:
        snapshot_path.write_text(
            state.model_copy(update={"in_progress_agents": dict(in_progress)}).model_dump_json(indent=2),
            encoding="utf-8",
        )

    tqdm.write(f"\n=== Round {round_idx}/{state.n_rounds - 1} ===")
    if in_progress:
        tqdm.write(f"round {round_idx}: resuming -- {len(in_progress)} agent(s) already done this round")

    agent_docs: list[list[Document]] = []
    review = None
    if round_idx == 0 and remaining:
        # First round: no prior result to react to -- every agent starts from the same
        # dataset-derived query (RETRIEVAL_QUERY, "how do I map this kind of target given
        # this kind of dataset"), but retrieve_diverse shards one shared candidate pool
        # round-robin across agents so they don't all ground themselves in the same
        # top-k papers.
        tqdm.write(f"round {round_idx}: retrieving literature from the corpus for {len(state.agents)} agent(s)...")
        agent_docs = retrieve_diverse(RETRIEVAL_QUERY, settings, len(state.agents))
    elif remaining:
        # Call A: one peer review for the whole swarm, from last round's validation predictions.
        tqdm.write(f"round {round_idx}: peer review of round {round_idx - 1}...")
        review = peer_review.run_peer_review(peer_review_llm, state, settings, settings.runs_dir)

    agent_bar = tqdm(remaining, desc=f"round {round_idx}", unit="agent")

    # Fixed random subset of agents that always get an enriched reflection. Seeded from
    # settings.seed so the same agents are picked every round (and after a resume).
    n_random_enriched = 0
    if settings.random_enriched_reflection > 0:
        n_random_enriched = max(1, round(settings.random_enriched_reflection * len(state.agents)))
    random_enriched_agents = set(
        random.Random(settings.seed).sample([a.agent_idx for a in state.agents], min(n_random_enriched, len(state.agents)))
    )

    for agent in agent_bar:
        tag = f"[round {round_idx} | agent {agent.agent_idx}]"
        agent_bar.set_postfix_str(f"agent {agent.agent_idx}")
        agent_dir = artifacts.agent_round_dir(
            settings.runs_dir, state.run_id, round_idx, agent.agent_idx
        )
        changed_lines = None

        if round_idx == 0:
            docs = agent_docs[agent.agent_idx]
            tqdm.write(f"{tag} retrieved {len(docs)} paper(s); writing initial pipeline.py...")
            code = generate_initial_pipeline(code_llm, format_for_prompt(docs, settings))
            velocity = agent.velocity
            velocity_history = agent.velocity_history
            retrieved_papers = [d.metadata.get("paper_id", "unknown") for d in docs]
        else:
            enriched_reflection = (
                settings.enriched_reflection_all
                or agent.last_dice == state.g_best_score
                or agent.agent_idx in random_enriched_agents
            )
            # Later rounds: Reflect (-> GroundReflection) -> VelocityUpdate -> ApplyVelocity
            # against this round's peer review -- see peer_review.py.
            tqdm.write(
                f"{tag} peer review: reflect -> velocity -> pipeline"
                f"{' (enriched reflection)' if enriched_reflection else ''}..."
            )
            update = peer_review.reflect_and_update(
                llm, code_llm, agent, review, state.g_best_code, state.g_best_score,
                settings, round_idx, enriched_reflection,
            )
            code, velocity, velocity_history = update.code, update.velocity, update.velocity_history
            retrieved_papers, changed_lines = update.retrieved_papers, update.changed_lines
            artifacts.write_text(agent_dir, "reflection.md", update.reflection)
            artifacts.write_text(agent_dir, "velocity.md", update.velocity)

        cited_papers = peer_review.extract_cited_papers(code)

        tqdm.write(f"{tag} running pipeline (train + evaluate, up to {settings.max_debug_iters} debug iters)...")
        result = run_agent_round(
            agent_dir,
            code,
            code_llm,
            data_npz_path,
            settings.max_debug_iters,
            settings.exec_timeout_s,
            seeds,
        )

        metrics = None
        if result.success:
            tqdm.write(
                f"{tag} done: Dice={result.dice:.4f} ± {result.dice_std:.4f} over {len(seeds)} seed(s) "
                f"IoU={result.iou:.4f} (debug iters: {result.debug_iters})"
            )
            preds = np.load(artifacts.val_preds_path(agent_dir, seeds[0]))["preds"]
            full = observe.agent_metrics(preds, y_val, size_edges)
            metrics = {k: full[k] for k in _LEDGER_METRICS}
        else:
            tqdm.write(f"{tag} FAILED after {result.debug_iters} debug iter(s): {result.error}")

        score = result.dice if result.success else float("-inf")
        new_agent = agent.model_copy(
            update={
                # Position moves only to a pipeline that ran; a failed attempt leaves
                # the last working one to revise (unless there has never been one).
                "pipeline_code": result.final_code if (result.success or not agent.pipeline_code) else agent.pipeline_code,
                "velocity": velocity,
                "velocity_history": velocity_history,
                "last_dice": result.dice,
                "last_dice_per_seed": result.dice_per_seed,
                "last_dice_std": result.dice_std,
                "last_iou": result.iou,
                "last_hparams": result.hparams if result.success else agent.last_hparams,
                "last_error": None if result.success else result.error,
            }
        )
        if score > new_agent.p_best_score + settings.p_best_epsilon:
            tqdm.write(f"{tag} new personal-best (Dice={score:.4f})")
            new_agent = new_agent.model_copy(
                update={"p_best_code": result.final_code, "p_best_score": score}
            )

        _append_ledger(
            settings.runs_dir,
            state.run_id,
            {
                "round": round_idx,
                "agent_idx": agent.agent_idx,
                "dice": result.dice,  # mean over the evaluation seeds
                "dice_per_seed": result.dice_per_seed,
                "dice_std": result.dice_std,
                "iou": result.iou,
                "success": result.success,
                "error": result.error,
                "retrieved_papers": retrieved_papers,
                "cited_papers": cited_papers,
                "debug_iters": result.debug_iters,
                # How big the move was, to audit that a velocity really was one change.
                "changed_lines": changed_lines,
                "hparams": result.hparams,
                "hparams_changed": (
                    peer_review.changed_hparams(agent.last_hparams, result.hparams)
                    if result.success and round_idx > 0 else []
                ),
                "metrics": metrics,
            },
        )
        in_progress[agent.agent_idx] = new_agent
        _checkpoint()

    updated_agents = [in_progress[a.agent_idx] for a in state.agents]

    g_best_code, g_best_score = state.g_best_code, state.g_best_score
    for agent in updated_agents:
        if agent.p_best_score > g_best_score + settings.p_best_epsilon:
            g_best_code, g_best_score = agent.p_best_code, agent.p_best_score

    if g_best_score > state.g_best_score + settings.p_best_epsilon:
        tqdm.write(f"=== Round {round_idx} new global-best: Dice={g_best_score:.4f} ===")
        rounds_without_improvement = 0
    else:
        rounds_without_improvement = state.rounds_without_improvement + 1
        tqdm.write(
            f"=== Round {round_idx}: no global-best improvement "
            f"({rounds_without_improvement}/{settings.plateau_patience} before early stop) ==="
        )

    updates = {
        "round": round_idx + 1,
        "agents": updated_agents,
        "g_best_code": g_best_code,
        "g_best_score": g_best_score,
        "rounds_without_improvement": rounds_without_improvement,
        "in_progress_agents": {},
    }

    # Single checkpoint file, overwritten after every agent (not just every round):
    # always holds exactly the state as of the last agent to finish -- this is the only
    # thing a resumed run reads, whether that's mid-round or a cleanly completed one.
    snapshot_path.write_text(
        state.model_copy(update=updates).model_dump_json(indent=2), encoding="utf-8"
    )

    return updates


def _finalize_node(state: SwarmState, settings: Settings) -> dict:
    stop_reason = (
        "plateau"
        if state.rounds_without_improvement >= settings.plateau_patience
        else "max_rounds"
    )
    tqdm.write(f"\n=== Stopping after round {state.round - 1} (reason: {stop_reason}) ===")

    # Per-round history, including the citation trail (which papers were retrieved vs.
    # actually cited in the pipeline's comments each round).
    ledger_entries = _read_ledger(settings.runs_dir, state.run_id)
    per_agent_history = {}
    for a in state.agents:
        rounds = sorted(
            (e for e in ledger_entries if e["agent_idx"] == a.agent_idx),
            key=lambda e: e["round"],
        )
        per_agent_history[str(a.agent_idx)] = {
            "rounds": [
                {
                    "round": e["round"],
                    "dice": e.get("dice"),
                    "dice_per_seed": e.get("dice_per_seed"),
                    "dice_std": e.get("dice_std"),
                    "iou": e.get("iou"),
                    "success": e.get("success"),
                    "debug_iters": e.get("debug_iters", 0),
                    "changed_lines": e.get("changed_lines"),
                    "hparams_changed": e.get("hparams_changed", []),
                    "retrieved_papers": e.get("retrieved_papers", []),
                    "cited_papers": e.get("cited_papers", []),
                }
                for e in rounds
            ],
        }

    best_agent = None
    for a in state.agents:
        if a.p_best_score == state.g_best_score:
            best_agent = a.agent_idx
            break

    summary = {
        "run_id": state.run_id,
        "git": git_version(),
        "llms": {
            "primary": {"provider": settings.llm_provider, "model": settings.model_name},
            "peer_review": {"provider": settings.llm_provider_2, "model": settings.model_name_2},
            "code": {"provider": settings.llm_provider_3, "model": settings.model_name_3},
        },
        "stop_reason": stop_reason,
        "eval_seeds": settings.active_seeds(),
        "rounds_run": state.round,
        "g_best_score": state.g_best_score,
        "best_agent": best_agent,
        "per_agent_best": [
            {
                "agent_idx": a.agent_idx,
                "p_best_score": a.p_best_score,
            }
            for a in state.agents
        ],
        "per_agent_history": per_agent_history,
    }
    if state.g_best_code:
        (settings.runs_dir / state.run_id / "g_best_pipeline.py").write_text(
            state.g_best_code, encoding="utf-8"
        )
    summary_path = settings.runs_dir / state.run_id / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # One row per run in <runs_dir>/registry.{jsonl,csv}: config + outcome, comparable across runs.
    registry.register_run(
        settings,
        registry.build_record(settings, state, stop_reason, ledger_entries, summary["git"]),
    )
    return {}


def _should_continue(state: SwarmState, settings: Settings) -> str:
    if state.round >= state.n_rounds:
        return "finalize"
    if state.rounds_without_improvement >= settings.plateau_patience:
        return "finalize"
    return "continue"


def build_graph(settings: Settings):
    graph = StateGraph(SwarmState)
    graph.add_node("run_round", lambda s: _run_round_node(s, settings))
    graph.add_node("finalize", lambda s: _finalize_node(s, settings))
    graph.set_entry_point("run_round")
    graph.add_conditional_edges(
        "run_round",
        lambda s: _should_continue(s, settings),
        {"continue": "run_round", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
