"""Small helpers shared across modules that don't warrant their own package."""
from __future__ import annotations

import csv
import re
import subprocess
from functools import lru_cache
from pathlib import Path

# Input data Landslide4Sense lacks (it's single-date optical + DEM): a paper whose method needs any
# of these can't be reproduced on it. Same patterns as notebooks/all_runs_overview.ipynb
# (fig_corpus_applicability), so the filter and the report figure agree.
_NEEDS_DATA_L4S_LACKS = [
    re.compile(r"\b(SAR|InSAR|Sentinel-1|interferometr\w*|radar)\b"),       # case-sensitive: "SAR" acronym
    re.compile(r"\b(LiDAR|point cloud)\b", re.I),
    re.compile(r"(change detection|pre-event|post-event|pre- and post|bi-?temporal|multi-?temporal|time[ -]series)", re.I),
]


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


@lru_cache(maxsize=None)
def l4s_usable_paper_ids(metadata_csv: Path, json_dir: Path) -> frozenset[str]:
    """paper_ids (Scopus EIDs) whose method runs on Landslide4Sense-like input: papers whose
    JSON summary, title, abstract, datasets or methods mention SAR/InSAR, LiDAR or
    multi-temporal/change-detection data are dropped. Papers the human review left without a
    method class are dropped too. Cached, since the CSV and JSONs don't change within a run."""
    with open(metadata_csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if r["method_class_name"]]
    usable = set()
    for r in rows:
        p = json_dir / f"{r['EID']}.json"
        text = " ".join([p.read_text(encoding="utf-8") if p.exists() else "",
                         r["Title"], r["Abstract"], r["Datasets"], r["Methods"]])
        if not any(pat.search(text) for pat in _NEEDS_DATA_L4S_LACKS):
            usable.add(r["EID"])
    return frozenset(usable)
