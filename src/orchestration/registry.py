"""Results registry: one row per finished run, kept across runs in <runs_dir>.

`registry.jsonl` is the full record (append/upsert by run_id, written at the end of every
run -- including a run that stopped on plateau or was resumed); `registry.csv` is the
same table flattened for spreadsheets. Per-agent, per-round detail stays in each run's
own ledger.jsonl; the registry answers "which configuration produced which result".

    from orchestration.registry import load_registry
    df = load_registry(settings.runs_dir)   # one row per run
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from agents.prompts import PROMPTS_PATH, PROMPTS_SHA
from config import Settings
from orchestration.state import SwarmState

JSONL_NAME = "registry.jsonl"
CSV_NAME = "registry.csv"


def llms(settings: Settings) -> dict:
    return {
        "agents": [{"n_agents": g.n_agents, "model": g.label} for g in settings.agents],
        "observer": f"{settings.llm_provider_2}/{settings.model_name_2}",
        "checker": f"{settings.llm_provider_3}/{settings.model_name_3}",
    }


def build_record(
    settings: Settings,
    state: SwarmState,
    stop_reason: str,
    rounds_run: int,
    ledger: list[dict],
    conversation: list[dict],
    git: dict | None,
) -> dict:
    """Everything worth comparing runs by: configuration, outcome, and the trajectory."""
    ok = [e for e in ledger if e.get("dice") is not None]
    best_row = max(ok, key=lambda e: e["dice"]) if ok else None

    curve, swarm_mean, best_so_far = [], [], float("-inf")
    for t in sorted({e["round"] for e in ledger}):
        scores = [e["dice"] for e in ok if e["round"] == t]
        if scores:
            best_so_far = max(best_so_far, max(scores))
            swarm_mean.append(round(sum(scores) / len(scores), 6))
        else:
            swarm_mean.append(None)
        curve.append(round(best_so_far, 6) if best_so_far > float("-inf") else None)

    posts_per_round = [
        sum(e.get("posts", 0) for e in conversation if e["round"] == t)
        for t in sorted({e["round"] for e in conversation})
    ]

    return {
        "run_id": state.run_id,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git,
        "dataset": settings.dataset_name,
        "prompts_file": PROMPTS_PATH.name,
        "prompts_sha": PROMPTS_SHA,  # which exact prompt text produced this result
        "llms": llms(settings),
        "n_agents": len(state.agents),
        "n_rounds_requested": settings.n_rounds,
        "rounds_run": rounds_run,
        "stop_reason": stop_reason,
        "eval_seeds": settings.active_seeds(),
        "p_best_epsilon": settings.p_best_epsilon,
        "plateau_patience": settings.plateau_patience,
        "use_rag": settings.use_rag,
        "web_search": settings.web_search,
        "observation_detail": settings.observation_detail,
        "retrieval_strategy": settings.retrieval_strategy,
        "retrieval_k": settings.retrieval_k,
        "retrieval_pool_size": settings.retrieval_pool_size,
        "mmr_lambda": settings.mmr_lambda,
        "retrieval_filter": settings.retrieval_filter,
        "conversation_cycles": settings.conversation_cycles,
        "max_checks_per_round": settings.max_checks_per_round,
        "agent_max_tool_calls": settings.agent_max_tool_calls,
        "max_train_tiles": settings.max_train_tiles,
        "g_best_score": state.g_best_score if state.g_best_score > float("-inf") else None,
        "g_best_dice_per_seed": best_row.get("dice_per_seed") if best_row else None,
        "g_best_dice_std": best_row.get("dice_std") if best_row else None,
        "best_agent": best_row["agent_idx"] if best_row else None,
        "best_round": best_row["round"] if best_row else None,
        "n_runs": len(ledger),
        "n_failed_runs": len(ledger) - len(ok),
        "g_best_curve": curve,  # best mean Dice so far, per round
        "swarm_mean_curve": swarm_mean,  # mean Dice of the agents that ran, per round
        "posts_per_round": posts_per_round,
        "n_checks": sum(e.get("checks_requested", 0) for e in conversation),
        "n_passes": sum(e.get("passes", 0) for e in conversation),
        "tool_calls": {
            "build": sum(e.get("build_tool_calls", 0) for e in ledger),
            "conversation": sum(e.get("tool_calls", 0) for e in conversation),
        },
    }


def register_run(settings: Settings, record: dict) -> Path:
    """Upsert `record` (by run_id) into registry.jsonl and rewrite registry.csv."""
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    jsonl = settings.runs_dir / JSONL_NAME

    records: list[dict] = []
    if jsonl.exists():
        records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["run_id"] != record["run_id"]] + [record]
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    _flatten(records).to_csv(settings.runs_dir / CSV_NAME, index=False)
    return jsonl


def _flatten(records: list[dict]) -> pd.DataFrame:
    """One flat row per run: nested dicts become dotted columns, lists become JSON text."""
    rows = []
    for r in records:
        flat = pd.json_normalize(r, sep=".").iloc[0].to_dict()
        rows.append({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in flat.items()})
    return pd.DataFrame(rows)


def load_registry(runs_dir: Path) -> pd.DataFrame:
    """The registry as a DataFrame (one row per run, oldest first); empty if none yet."""
    jsonl = runs_dir / JSONL_NAME
    if not jsonl.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _flatten(records)
