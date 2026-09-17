"""Dataset-agnostic entry points: dispatch to the active dataset's DatasetSpec
(config.settings.dataset_name, looked up in data/registry.py) plus the one bit of
splitting logic every id-indexed dataset shares.

Per-dataset loading logic (file layout, formats, id schemes) lives under
data/datasets/ -- this module never hardcodes any of that.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from config import settings
from data.spec import Split


def split_ids_by_ratio(
    ids: list[int], seed: int, ratios: dict[str, float]
) -> dict[Split, list[int]]:
    """Deterministically shuffle `ids` and carve train/val/test out by `ratios`.

    Shared by any id-indexed dataset loader (data/datasets/*.py) that needs a fixed,
    reproducible split -- the shuffle/cut logic is generic, only the id list itself is
    dataset-specific.
    """
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def load_split(
    split: Split,
    dataset_dir: Path,
    max_tiles: int | None = None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load a fixed split as stacked arrays for the active (or given) dataset."""
    from data.registry import get_dataset_spec

    spec = get_dataset_spec(dataset_name or settings.dataset_name)
    return spec.load_split(split, dataset_dir, max_tiles)


def load_smoke_sample(
    dataset_dir: Path, tile_ids: list[int], dataset_name: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load the fixed smoke-test tiles (by id) for the active (or given) dataset."""
    from data.registry import get_dataset_spec

    spec = get_dataset_spec(dataset_name or settings.dataset_name)
    return spec.load_smoke_sample(dataset_dir, tile_ids)


def build_data_cache(
    dataset_name: str,
    dataset_dir: Path,
    cache_path: Path,
    max_train_tiles: int,
    max_val_tiles: int,
    smoke_tile_ids: list[int] | None = None,
) -> Path:
    """Build the shared train/val (+fixed smoke sample) npz reused by every agent and
    round in a run, so the dataset is only ever loaded from disk once per run.

    smoke_tile_ids defaults to the dataset's own DatasetSpec.smoke_tile_ids; pass it
    explicitly only to override.
    """
    from data.registry import get_dataset_spec

    spec = get_dataset_spec(dataset_name)
    tile_ids = smoke_tile_ids if smoke_tile_ids is not None else spec.smoke_tile_ids

    X_train, y_train, _ = spec.load_split("train", dataset_dir, max_train_tiles)
    X_val, y_val, _ = spec.load_split("val", dataset_dir, max_val_tiles)
    X_sample, y_sample = spec.load_smoke_sample(dataset_dir, tile_ids)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        X_sample=X_sample,
        y_sample=y_sample,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    return cache_path
