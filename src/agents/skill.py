"""Per-run/round/agent storage: runs/<run_id>/round_<t>/agent_<i>/{skill.md,pipeline.py}."""
from __future__ import annotations

from pathlib import Path


def agent_round_dir(runs_dir: Path, run_id: str, round_idx: int, agent_idx: int) -> Path:
    d = runs_dir / run_id / f"round_{round_idx}" / f"agent_{agent_idx}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_skill(agent_dir: Path, content: str) -> Path:
    path = agent_dir / "skill.md"
    path.write_text(content, encoding="utf-8")
    return path


def read_skill(agent_dir: Path) -> str | None:
    path = agent_dir / "skill.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def write_pipeline(agent_dir: Path, code: str) -> Path:
    path = agent_dir / "pipeline.py"
    path.write_text(code, encoding="utf-8")
    return path


def pipeline_path(agent_dir: Path) -> Path:
    return agent_dir / "pipeline.py"


def previous_skill(
    runs_dir: Path, run_id: str, round_idx: int, agent_idx: int
) -> str | None:
    """The same agent's skill.md from the previous round, or None at round 0."""
    if round_idx == 0:
        return None
    prev_dir = runs_dir / run_id / f"round_{round_idx - 1}" / f"agent_{agent_idx}"
    return read_skill(prev_dir)
