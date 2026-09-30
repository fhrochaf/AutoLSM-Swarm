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

from config import Settings
from orchestration.state import SwarmState

JSONL_NAME = "registry.jsonl"
CSV_NAME = "registry.csv"


def build_record(
    settings: Settings,
    state: SwarmState,
    stop_reason: str,
    ledger: list[dict],
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

    return {
        "run_id": state.run_id,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git,
        "dataset": settings.dataset_name,
        "llms": {
            "primary": f"{settings.llm_provider}/{settings.model_name}",
            "peer_review": f"{settings.llm_provider_2}/{settings.model_name_2}",
            "code": f"{settings.llm_provider_3}/{settings.model_name_3}",
        },
        "n_agents": len(state.agents),
        "n_rounds_requested": state.n_rounds,
        "rounds_run": state.round,
        "stop_reason": stop_reason,
        "eval_seeds": settings.active_seeds(),
        "p_best_epsilon": settings.p_best_epsilon,
        "plateau_patience": settings.plateau_patience,
        "velocity_history_len": settings.velocity_history_len,
        "enriched_reflection_all": settings.enriched_reflection_all,
        "random_enriched_reflection": settings.random_enriched_reflection,
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
