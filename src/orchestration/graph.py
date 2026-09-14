"""The round loop, wired as a LangGraph StateGraph.

Round 0 seeds every agent from the literature corpus and generates its first
pipeline.py. Later rounds ask peer_review to revise each agent's skill against the
swarm's global-best and its own personal-best; when there are no peers to reflect
against (n_agents == 1), that agent's previous round is simply carried forward instead
of wastefully re-running an identical pipeline.

SwarmState/AgentState are pydantic models (orchestration/state.py). LangGraph passes an
actual model instance into each node (attribute access below), and node functions
return a plain dict of the fields that changed -- LangGraph merges that back into the
persisted state. `graph.invoke(...)` itself always returns a plain dict regardless of
the schema type, not a model instance.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from tqdm import tqdm

from agents import skill as skill_io
from agents.codegen import generate_pipeline, write_skill
from agents.prompts import RETRIEVAL_QUERY
from agents.runner import run_agent_round
from config import Settings
from corpus.retrieve import format_for_prompt, retrieve_diverse
from orchestration import peer_review
from orchestration.state import AgentState, SwarmState


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
    already holds every agent's current velocity/p_best/skill/code and the swarm's
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


def _run_round_node(state: SwarmState, settings: Settings) -> dict:
    llm = init_chat_model(
        model=settings.model_name,
        model_provider=settings.llm_provider,
        api_key=settings.api_key or None,
        # temperature=settings.temperature,
        max_tokens=16000,
    )

    judge_llm = init_chat_model(
        model=settings.model_name_2,
        model_provider=settings.llm_provider_2,
        api_key=settings.api_key_2 or None,
        # temperature=settings.temperature,
        max_tokens=2000,
    )

    # Used only for pipeline.py creation (generate_pipeline/fix_pipeline/
    # revise_pipeline_for_fidelity) -- kept separate from the primary model above.
    code_llm = init_chat_model(
        model=settings.model_name_3,
        model_provider=settings.llm_provider_3,
        api_key=settings.api_key_3 or None,
        # temperature=settings.temperature,
        max_tokens=16000,
    )
    round_idx = state.round
    data_npz_path = Path(state.data_npz_path)
    snapshot_path = settings.runs_dir / state.run_id / "state_snapshot.json"

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
    agent_bar = tqdm(remaining, desc=f"round {round_idx}", unit="agent")

    agent_docs: list[list[Document]] = []
    if round_idx == 0 and remaining:
        # First round: no prior result to react to -- every agent starts from the same
        # dataset-derived query (RETRIEVAL_QUERY, "how do I map this kind of target given
        # this kind of dataset"), but retrieve_diverse shards one shared candidate pool
        # round-robin across agents so they don't all ground themselves in the same
        # top-k papers.
        tqdm.write(f"round {round_idx}: retrieving literature from the corpus for {len(state.agents)} agent(s)...")
        agent_docs = retrieve_diverse(RETRIEVAL_QUERY, settings, len(state.agents))

    for agent in agent_bar:
        tag = f"[round {round_idx} | agent {agent.agent_idx}]"
        agent_bar.set_postfix_str(f"agent {agent.agent_idx}")
        agent_dir = skill_io.agent_round_dir(
            settings.runs_dir, state.run_id, round_idx, agent.agent_idx
        )

        if round_idx == 0:
            docs = agent_docs[agent.agent_idx]
            tqdm.write(f"{tag} retrieved {len(docs)} paper(s); writing initial skill.md...")
            skill_md = write_skill(llm, format_for_prompt(docs, settings))
            velocity = agent.velocity
            skill_changed = True
            retrieved_papers = [d.metadata.get("paper_id", "unknown") for d in docs]
        else:
            # Later rounds: peer review (Reflect -> VelocityUpdate -> SkillUpdate)
            # decides whether/how the skill changes -- see peer_review.py. Reflect is
            # grounded in a fresh corpus retrieval every round, not just round 0.
            neighbourhood = [a for a in state.agents if a.agent_idx != agent.agent_idx]
            tqdm.write(f"{tag} peer review: reflecting against {len(neighbourhood)} peer(s)...")
            skill_md, velocity, skill_changed, retrieved_papers = peer_review.reflect_and_update(
                llm, agent, neighbourhood, state.g_best_skill, settings
            )

        cited_papers = peer_review.extract_cited_papers(skill_md)

        if not skill_changed:
            tqdm.write(f"{tag} skill unchanged -> carrying previous round forward (no re-run)")
            prev_dir = settings.runs_dir / state.run_id / f"round_{round_idx - 1}" / f"agent_{agent.agent_idx}"
            for name in ("skill.md", "pipeline.py", "driver_stdout.json"):
                src_path = prev_dir / name
                if src_path.exists():
                    shutil.copy2(src_path, agent_dir / name)
            new_agent = agent.model_copy(update={"skill_md": skill_md, "last_skill_changed": False})
            _append_ledger(
                settings.runs_dir,
                state.run_id,
                {
                    "round": round_idx,
                    "agent_idx": agent.agent_idx,
                    "dice": agent.last_dice,
                    "iou": agent.last_iou,
                    "success": agent.last_dice is not None,
                    "error": None,
                    "carried_forward": True,
                    "debug_iters": 0,
                    "retrieved_papers": retrieved_papers,
                    "cited_papers": cited_papers,
                },
            )
            in_progress[agent.agent_idx] = new_agent
            _checkpoint()
            continue

        skill_io.write_skill(agent_dir, skill_md)
        tqdm.write(f"{tag} skill.md updated; generating pipeline.py...")
        code = generate_pipeline(code_llm, skill_md)

        tqdm.write(f"{tag} running pipeline (train + evaluate, up to {settings.max_debug_iters} debug iters)...")
        result = run_agent_round(
            agent_dir,
            skill_md,
            code,
            code_llm,
            judge_llm,
            data_npz_path,
            settings.max_debug_iters,
            settings.exec_timeout_s,
        )

        if result.success:
            tqdm.write(
                f"{tag} done: Dice={result.dice:.4f} IoU={result.iou:.4f} "
                f"(debug iters: {result.debug_iters})"
            )
        else:
            tqdm.write(f"{tag} FAILED after {result.debug_iters} debug iter(s): {result.error}")

        score = result.dice if result.success else float("-inf")
        new_agent = agent.model_copy(
            update={
                "skill_md": skill_md,
                "velocity": velocity,
                "last_dice": result.dice,
                "last_iou": result.iou,
                "last_code": result.final_code if result.success else "",
                "last_skill_changed": True,
            }
        )
        if score > new_agent.p_best_score + settings.p_best_epsilon:
            tqdm.write(f"{tag} new personal-best (Dice={score:.4f})")
            new_agent = new_agent.model_copy(
                update={"p_best_skill": new_agent.skill_md, "p_best_score": score}
            )

        _append_ledger(
            settings.runs_dir,
            state.run_id,
            {
                "round": round_idx,
                "agent_idx": agent.agent_idx,
                "dice": result.dice,
                "iou": result.iou,
                "success": result.success,
                "error": result.error,
                "carried_forward": False,
                "retrieved_papers": retrieved_papers,
                "cited_papers": cited_papers,
                "debug_iters": result.debug_iters,
            },
        )
        in_progress[agent.agent_idx] = new_agent
        _checkpoint()

    updated_agents = [in_progress[a.agent_idx] for a in state.agents]

    g_best_skill, g_best_score = state.g_best_skill, state.g_best_score
    for agent in updated_agents:
        if agent.p_best_score > g_best_score + settings.p_best_epsilon:
            g_best_skill, g_best_score = agent.p_best_skill, agent.p_best_score

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
        "g_best_skill": g_best_skill,
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
    # actually cited each round -- "provenance ... from every adopted pipeline change
    # back to the specific paper that suggested it", per the implementation notes).
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
                    "iou": e.get("iou"),
                    "success": e.get("success"),
                    "carried_forward": e.get("carried_forward", False),
                    "debug_iters": e.get("debug_iters", 0),
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
        "llms": {
            "primary": {"provider": settings.llm_provider, "model": settings.model_name},
            "judge": {"provider": settings.llm_provider_2, "model": settings.model_name_2},
            "code": {"provider": settings.llm_provider_3, "model": settings.model_name_3},
        },
        "stop_reason": stop_reason,
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
    summary_path = settings.runs_dir / state.run_id / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
