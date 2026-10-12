"""The conversational swarm, wired as a LangGraph StateGraph.

    START -> seed -> build_0 -> ... -> build_{N-1} -> update_bests -> observer --(last round / plateau)--> finalize -> END
                       ^                                                  |
                       |                                                  v
                       +---(cycles done / all PASS)--- next_speaker <--> talk_i --(check request)--> hypothesis_checker
                                                            ^                                               |
                                                            +-----------------------------------------------+

    seed                (round 0) each agent gets its own papers from the corpus (corpus/retrieve.py)
    build_i             agent i writes its pipeline.py (create_agent with corpus + web search tools),
                        which is run and scored right away (agents/runner.py -> driver.py subprocess)
    update_bests        personal-best / global-best / plateau counter
    observer            metric tables of every agent + the OBSERVER's comparative report (shared)
    next_speaker        round-robin over the agents (order rotated every round), conversation_cycles
                        cycles; a cycle in which every agent PASSes ends the conversation early
    talk_i              agent i reads the shared + private context and posts one message to the
                        blackboard (or PASSes); its tool results stay private. If it called
                        request_hypothesis_check, control is handed to the checker
    hypothesis_checker  runs analysis scripts over the round's validation predictions and posts the
                        numbers to the blackboard
    finalize            summary.json, g_best_pipeline.py, run registry

The whole state is checkpointed to <run>/checkpoints.sqlite after every node (thread_id = run
id), so main.py --resume continues from exactly where a run stopped.
"""
from __future__ import annotations

import json
import sqlite3
import time
from functools import partial
from pathlib import Path

import numpy as np
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from tqdm import tqdm

from agents import artifacts, checker, context, swarm_agent
from agents.prompts import PROMPTS_PATH, RETRIEVAL_QUERY
from agents.runner import run_agent_round
from agents.text_utils import changed_hparams, extract_cited_papers
from config import Settings
from corpus.retrieve import format_for_prompt, retrieve_diverse
from eval import observe
from orchestration import blackboard, observer, registry
from orchestration.state import CHECKPOINT_TYPES, AgentState, PendingCheck, Post, SwarmState
from utils import git_version

LEDGER = "ledger.jsonl"  # one row per agent per round: the pipeline it built and how it scored
CONVERSATION_LEDGER = "conversation_ledger.jsonl"  # one row per agent per round: what it did in the conversation

# Compact per-run behavior figures stored in the ledger (the full tables live in each
# round's observer_tables.md), so a run can be analyzed without reloading predictions.
_LEDGER_METRICS = (
    "precision", "recall", "tile_precision", "tile_recall", "fp_tiles", "object_recall",
    "object_precision", "pred_pos_rate", "true_pos_rate",
)


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_ledger(runs_dir: Path, run_id: str, name: str = LEDGER) -> list[dict]:
    """Rows of a run's ledger, one per (round, agent): a node re-run after a resume appends
    its row again, and the later row wins."""
    path = runs_dir / run_id / name
    if not path.exists():
        return []
    rows: dict[tuple[int, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            rows[(e["round"], e["agent_idx"])] = e
    return [rows[k] for k in sorted(rows)]


def create_initial_state(run_id: str, models: list[str], data_npz_path: Path) -> SwarmState:
    """`models`: the "<provider>/<model>" label of every agent, in agent_idx order."""
    return SwarmState(
        run_id=run_id,
        data_npz_path=str(data_npz_path),
        agents=[AgentState(agent_idx=i, model=m) for i, m in enumerate(models)],
    )


def open_checkpointer(run_dir: Path) -> SqliteSaver:
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(run_dir / "checkpoints.sqlite", check_same_thread=False)
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def _with_agent(state: SwarmState, agent: AgentState) -> list[AgentState]:
    return [agent if a.agent_idx == agent.agent_idx else a for a in state.agents]


# --------------------------------------------------------------------------------------
# Pipelines: seed -> build_i -> update_bests
# --------------------------------------------------------------------------------------


def _seed_node(state: SwarmState, settings: Settings) -> dict:
    if not settings.use_rag:
        tqdm.write("seed: use_rag is off -- agents start without literature")
        return {}
    tqdm.write(f"seed: retrieving literature from the corpus for {len(state.agents)} agent(s)...")
    docs_per_agent = retrieve_diverse(RETRIEVAL_QUERY, settings, len(state.agents))
    agents = []
    for agent, docs in zip(state.agents, docs_per_agent):
        agent_dir = artifacts.agent_round_dir(settings.runs_dir, state.run_id, 0, agent.agent_idx)
        artifacts.write_text(agent_dir, "seed_papers.md", format_for_prompt(docs, settings))
        agents.append(agent.model_copy(update={"seed_paper_ids": [d.metadata.get("paper_id", "unknown") for d in docs]}))
    return {"agents": agents}


def _build_node(state: SwarmState, settings: Settings, agent_idx: int) -> dict:
    agent = state.agents[agent_idx]
    round_idx, n_agents = state.round, len(state.agents)
    tag = f"[round {round_idx} | agent {agent_idx} | {agent.model}]"
    agent_dir = artifacts.agent_round_dir(settings.runs_dir, state.run_id, round_idx, agent_idx)
    seeds = settings.active_seeds()
    if round_idx == 0:
        if agent_idx == 0:
            tqdm.write(f"\n=== Round 0/{settings.n_rounds - 1} ===")
        seed_path = agent_dir / "seed_papers.md"
        seed_papers = seed_path.read_text(encoding="utf-8") if seed_path.exists() else None
        messages = context.build_messages(state, agent_idx, "build_initial", settings, seed_papers)
    else:
        messages = context.build_messages(state, agent_idx, "build", settings)

    tqdm.write(f"{tag} writing pipeline.py...")
    build = swarm_agent.build_pipeline(agent_idx, n_agents, messages, settings)
    artifacts.write_text(agent_dir, "build_transcript.md", build.transcript)
    notebook = build.notebook or agent.notebook
    artifacts.write_text(agent_dir, "notebook.md", notebook)

    tqdm.write(
        f"{tag} running pipeline (train + evaluate, up to {settings.max_debug_iters} debug iters; "
        f"{build.stats['tool_calls']} tool call(s) while writing)..."
    )
    def fix(code: str, error: str) -> str:
        fixed, log = swarm_agent.fix_pipeline(agent_idx, n_agents, settings, code, error)
        with (agent_dir / "fix_transcript.md").open("a", encoding="utf-8") as f:
            f.write(log + "\n")
        return fixed

    result = run_agent_round(
        agent_dir,
        build.code,
        fix,
        Path(state.data_npz_path),
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
        y_val = np.load(state.data_npz_path)["y_val"]
        preds = np.load(artifacts.val_preds_path(agent_dir, seeds[0]))["preds"]
        full = observe.agent_metrics(preds, y_val, observe.size_edges(y_val))
        metrics = {k: full[k] for k in _LEDGER_METRICS}
    else:
        tqdm.write(f"{tag} FAILED after {result.debug_iters} debug iter(s): {result.error}")

    score = result.dice if result.success else float("-inf")
    new_agent = agent.model_copy(
        update={
            # The position moves only to a pipeline that ran; a failed attempt leaves the last
            # working one (unless there has never been one).
            "pipeline_code": result.final_code if (result.success or not agent.pipeline_code) else agent.pipeline_code,
            "last_dice": result.dice,
            "last_dice_per_seed": result.dice_per_seed,
            "last_dice_std": result.dice_std,
            "last_iou": result.iou,
            "last_hparams": result.hparams if result.success else agent.last_hparams,
            "last_error": None if result.success else result.error,
            "notebook": notebook,
        }
    )
    if score > new_agent.p_best_score + settings.p_best_epsilon:
        tqdm.write(f"{tag} new personal-best (Dice={score:.4f})")
        new_agent = new_agent.model_copy(update={"p_best_code": result.final_code, "p_best_score": score})

    _append_jsonl(
        settings.runs_dir / state.run_id / LEDGER,
        {
            "round": round_idx,
            "agent_idx": agent_idx,
            "model": agent.model,
            "dice": result.dice,  # mean over the evaluation seeds
            "dice_per_seed": result.dice_per_seed,
            "dice_std": result.dice_std,
            "iou": result.iou,
            "success": result.success,
            "error": result.error,
            "debug_iters": result.debug_iters,
            "seed_papers": agent.seed_paper_ids if round_idx == 0 else [],
            "cited_papers": extract_cited_papers(result.final_code),
            # What the agent looked up while writing this pipeline.
            **{f"build_{k}": v for k, v in build.stats.items()},
            "hparams": result.hparams,
            "hparams_changed": (
                changed_hparams(agent.last_hparams, result.hparams) if result.success and round_idx > 0 else []
            ),
            "metrics": metrics,
        },
    )
    return {"agents": _with_agent(state, new_agent)}


def _update_bests_node(state: SwarmState, settings: Settings) -> dict:
    g_best_code, g_best_score, g_best_agent = state.g_best_code, state.g_best_score, state.g_best_agent
    for agent in state.agents:
        if agent.p_best_score > g_best_score + settings.p_best_epsilon:
            g_best_code, g_best_score, g_best_agent = agent.p_best_code, agent.p_best_score, agent.agent_idx

    if g_best_score > state.g_best_score + settings.p_best_epsilon:
        tqdm.write(f"=== Round {state.round} new global-best: Dice={g_best_score:.4f} (agent_{g_best_agent}) ===")
        rounds_without_improvement = 0
    else:
        rounds_without_improvement = state.rounds_without_improvement + 1
        tqdm.write(
            f"=== Round {state.round}: no global-best improvement "
            f"({rounds_without_improvement}/{settings.plateau_patience} before early stop) ==="
        )
    return {
        "g_best_code": g_best_code,
        "g_best_score": g_best_score,
        "g_best_agent": g_best_agent,
        "rounds_without_improvement": rounds_without_improvement,
    }


# --------------------------------------------------------------------------------------
# Observation and conversation: observer -> next_speaker <-> talk_i (-> hypothesis_checker)
# --------------------------------------------------------------------------------------


def _observer_node(state: SwarmState, settings: Settings) -> dict:
    tqdm.write(f"round {state.round}: OBSERVER comparing the agents...")
    report, tables = observer.run_observer(state, settings)
    n = len(state.agents)
    shift = state.round % n
    # A fresh conversation: the previous round's board was kept until now because the agents
    # read it while writing this round's pipelines.
    return {
        "observer_report": report,
        "observer_tables": tables,
        "blackboard": [],
        "turn_order": list(range(shift, n)) + list(range(shift)),
        "cycle": 0,
        "speaker_ptr": 0,
        "passes_this_cycle": 0,
        "pending_check": None,
        "checks_used": 0,
        "talk_stats": {},
    }


def _route_after_observer(state: SwarmState, settings: Settings) -> str:
    if state.round + 1 >= settings.n_rounds:
        return "finalize"
    if state.rounds_without_improvement >= settings.plateau_patience:
        return "finalize"
    return "next_speaker"


def _empty_talk_stats() -> dict:
    return {"posts": 0, "passes": 0, "checks_requested": 0, "tool_calls": 0,
            "rag_queries": [], "rag_papers": [], "web_queries": []}


def _end_conversation(state: SwarmState, settings: Settings, cycle: int, reason: str) -> Command:
    tqdm.write(f"round {state.round}: conversation over ({reason}); {len(state.blackboard)} post(s) on the blackboard")
    for agent in state.agents:
        _append_jsonl(
            settings.runs_dir / state.run_id / CONVERSATION_LEDGER,
            {"round": state.round, "agent_idx": agent.agent_idx,
             **state.talk_stats.get(agent.agent_idx, _empty_talk_stats())},
        )
    tqdm.write(f"\n=== Round {state.round + 1}/{settings.n_rounds - 1} ===")
    return Command(update={"round": state.round + 1, "cycle": cycle}, goto="build_0")


def _next_speaker_node(state: SwarmState, settings: Settings) -> Command:
    n = len(state.agents)
    ptr, cycle, passes = state.speaker_ptr, state.cycle, state.passes_this_cycle
    if settings.conversation_cycles <= 0:
        return _end_conversation(state, settings, cycle, "conversation_cycles is 0")
    if ptr >= n:  # a full cycle is done
        all_passed = passes >= n
        cycle, ptr, passes = cycle + 1, 0, 0
        if all_passed:
            return _end_conversation(state, settings, cycle, "every agent passed")
        if cycle >= settings.conversation_cycles:
            return _end_conversation(state, settings, cycle, f"{settings.conversation_cycles} cycle(s) done")
    speaker = state.turn_order[ptr]
    return Command(
        update={"cycle": cycle, "speaker_ptr": ptr, "passes_this_cycle": passes},
        goto=f"talk_{speaker}",
    )


def _talk_node(state: SwarmState, settings: Settings, agent_idx: int) -> Command:
    round_idx, cycle = state.round, state.cycle
    tag = f"[round {round_idx} | cycle {cycle + 1} | agent {agent_idx}]"
    checks_left = max(0, settings.max_checks_per_round - state.checks_used)
    messages = context.build_messages(state, agent_idx, "talk", settings)
    turn = swarm_agent.take_turn(agent_idx, len(state.agents), messages, settings, checks_left)

    agent_dir = artifacts.agent_round_dir(settings.runs_dir, state.run_id, round_idx, agent_idx)
    with (agent_dir / "talk_transcript.md").open("a", encoding="utf-8") as f:
        f.write(f"\n# Cycle {cycle + 1}\n\n{turn.transcript}")

    board = list(state.blackboard)
    stats = dict(state.talk_stats.get(agent_idx, _empty_talk_stats()))
    stats["tool_calls"] += turn.stats["tool_calls"]
    for k in ("rag_queries", "rag_papers", "web_queries"):
        stats[k] = stats[k] + turn.stats[k]

    if turn.post is not None:
        board.append(Post(round=round_idx, cycle=cycle, author=f"agent_{agent_idx}", text=turn.post))
        stats["posts"] += 1
    update: dict = {"speaker_ptr": state.speaker_ptr + 1}
    goto = "next_speaker"
    if turn.check_question is not None and checks_left > 0:
        board.append(Post(round=round_idx, cycle=cycle, author=f"agent_{agent_idx}", kind="check_request", text=turn.check_question))
        stats["checks_requested"] += 1
        update |= {
            "pending_check": PendingCheck(requester=agent_idx, question=turn.check_question),
            "checks_used": state.checks_used + 1,
        }
        goto = "hypothesis_checker"
    passed = turn.post is None and goto == "next_speaker"
    if passed:
        stats["passes"] += 1
    tqdm.write(f"{tag} {'PASS' if passed else 'posted'}{' + check request' if goto == 'hypothesis_checker' else ''}")

    blackboard.save_board(artifacts.round_dir(settings.runs_dir, state.run_id, round_idx), board)
    update |= {
        "blackboard": board,
        "passes_this_cycle": state.passes_this_cycle + int(passed),
        "talk_stats": {**state.talk_stats, agent_idx: stats},
    }
    return Command(update=update, goto=goto)


def _checker_node(state: SwarmState, settings: Settings) -> dict:
    pending = state.pending_check
    tqdm.write(f"round {state.round}: HYPOTHESIS CHECKER answering agent_{pending.requester}...")
    answer = checker.run_check(state, pending, settings)
    board = [
        *state.blackboard,
        Post(round=state.round, cycle=state.cycle, author="checker", kind="check_result",
             text=f"(answering agent_{pending.requester})\n{answer}"),
    ]
    blackboard.save_board(artifacts.round_dir(settings.runs_dir, state.run_id, state.round), board)
    return {"blackboard": board, "pending_check": None}


# --------------------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------------------


def _finalize_node(state: SwarmState, settings: Settings) -> dict:
    stop_reason = (
        "plateau" if state.rounds_without_improvement >= settings.plateau_patience else "max_rounds"
    )
    rounds_run = state.round + 1
    tqdm.write(f"\n=== Stopping after round {state.round} (reason: {stop_reason}) ===")

    ledger = read_ledger(settings.runs_dir, state.run_id)
    conversation = read_ledger(settings.runs_dir, state.run_id, CONVERSATION_LEDGER)
    per_agent_history = {}
    for a in state.agents:
        talk = {e["round"]: e for e in conversation if e["agent_idx"] == a.agent_idx}
        per_agent_history[str(a.agent_idx)] = [
            {
                "round": e["round"],
                "dice": e.get("dice"),
                "dice_per_seed": e.get("dice_per_seed"),
                "dice_std": e.get("dice_std"),
                "iou": e.get("iou"),
                "success": e.get("success"),
                "debug_iters": e.get("debug_iters", 0),
                "hparams_changed": e.get("hparams_changed", []),
                "cited_papers": e.get("cited_papers", []),
                "build_rag_papers": e.get("build_rag_papers", []),
                "conversation": {k: v for k, v in talk.get(e["round"], {}).items() if k not in ("round", "agent_idx")},
            }
            for e in ledger if e["agent_idx"] == a.agent_idx
        ]

    summary = {
        "run_id": state.run_id,
        "git": git_version(),
        "llms": registry.llms(settings),
        "stop_reason": stop_reason,
        "eval_seeds": settings.active_seeds(),
        "prompts_file": PROMPTS_PATH.name,
        "rounds_run": rounds_run,
        "conversation_cycles": settings.conversation_cycles,
        "g_best_score": state.g_best_score,
        "best_agent": state.g_best_agent,
        "per_agent_best": [
            {"agent_idx": a.agent_idx, "model": a.model, "p_best_score": a.p_best_score} for a in state.agents
        ],
        "per_agent_history": per_agent_history,
    }
    run_dir = settings.runs_dir / state.run_id
    if state.g_best_code:
        (run_dir / "g_best_pipeline.py").write_text(state.g_best_code, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # One row per run in <runs_dir>/registry.{jsonl,csv}: config + outcome, comparable across runs.
    registry.register_run(
        settings,
        registry.build_record(settings, state, stop_reason, rounds_run, ledger, conversation, summary["git"]),
    )
    return {"stop_reason": stop_reason}


# --------------------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------------------


def build_graph(settings: Settings, n_agents: int, checkpointer: SqliteSaver | None = None):
    graph = StateGraph(SwarmState)
    graph.add_node("seed", partial(_seed_node, settings=settings))
    for i in range(n_agents):
        graph.add_node(f"build_{i}", partial(_build_node, settings=settings, agent_idx=i))
    graph.add_node("update_bests", partial(_update_bests_node, settings=settings))
    graph.add_node("observer", partial(_observer_node, settings=settings))
    graph.add_node(
        "next_speaker", partial(_next_speaker_node, settings=settings),
        destinations=tuple(f"talk_{i}" for i in range(n_agents)) + ("build_0",),
    )
    for i in range(n_agents):
        graph.add_node(
            f"talk_{i}", partial(_talk_node, settings=settings, agent_idx=i),
            destinations=("next_speaker", "hypothesis_checker"),
        )
    graph.add_node("hypothesis_checker", partial(_checker_node, settings=settings))
    graph.add_node("finalize", partial(_finalize_node, settings=settings))

    graph.add_edge(START, "seed")
    graph.add_edge("seed", "build_0")
    for i in range(n_agents - 1):
        graph.add_edge(f"build_{i}", f"build_{i + 1}")
    graph.add_edge(f"build_{n_agents - 1}", "update_bests")
    graph.add_edge("update_bests", "observer")
    graph.add_conditional_edges(
        "observer", partial(_route_after_observer, settings=settings),
        {"next_speaker": "next_speaker", "finalize": "finalize"},
    )
    graph.add_edge("hypothesis_checker", "next_speaker")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def recursion_limit(settings: Settings, n_agents: int, rounds_left: int) -> int:
    """Upper bound on graph steps for `rounds_left` rounds (LangGraph counts every node run)."""
    per_round = (
        n_agents + 2  # builds, update_bests, observer
        + settings.conversation_cycles * 2 * n_agents + 1  # talk_i + next_speaker per turn
        + 2 * settings.max_checks_per_round  # checker + next_speaker per check
    )
    return per_round * max(rounds_left, 1) + 10
