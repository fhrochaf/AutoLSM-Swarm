"""Small helpers shared across modules that don't warrant their own package."""
from __future__ import annotations

import subprocess
from pathlib import Path


def git_version() -> dict:
    """Best-effort git commit + dirty-tree flag, so a run's summary.json records
    exactly which code produced it. Never raises -- a run shouldn't fail just
    because git is missing or this isn't a checkout (e.g. a packaged/CI environment).
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}
