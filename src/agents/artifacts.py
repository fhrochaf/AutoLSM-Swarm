"""Per-run/round/agent storage: runs/<run_id>/round_<t>/agent_<i>/{pipeline.py,
reflection.md, velocity.md, val_preds_seed<S>.npz, driver_stdout.json}, plus
round_<t>/peer_review.md (+ peer_review_tables.md) shared by the whole swarm."""
from __future__ import annotations

from pathlib import Path


def round_dir(runs_dir: Path, run_id: str, round_idx: int) -> Path:
    d = runs_dir / run_id / f"round_{round_idx}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_round_dir(runs_dir: Path, run_id: str, round_idx: int, agent_idx: int) -> Path:
    d = round_dir(runs_dir, run_id, round_idx) / f"agent_{agent_idx}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_pipeline(agent_dir: Path, code: str) -> Path:
    path = agent_dir / "pipeline.py"
    path.write_text(code, encoding="utf-8")
    return path


def write_text(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def val_preds_path(agent_dir: Path, seed: int) -> Path:
    return agent_dir / f"val_preds_seed{seed}.npz"
