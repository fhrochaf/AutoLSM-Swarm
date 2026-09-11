"""Fixed train/val/test split + loading for the labeled landslide tiles.
We carve our own train/val/test split out of the labeled tiles, once, with a fixed seed.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Literal
from config import Settings

import h5py
import numpy as np

Split = Literal["train", "val", "test"]

# Settings fields, not the class itself -- pydantic model classes don't expose field
# defaults as plain class attributes, so this must read from an instance.
settings = Settings()
SPLIT_SEED = settings.seed
SPLIT_RATIOS = settings.split_ratios


def _labeled_ids(dataset_dir: Path) -> list[int]:
    img_dir = dataset_dir / "TrainData" / "img"
    mask_dir = dataset_dir / "TrainData" / "mask"
    ids = []
    for p in img_dir.glob("image_*.h5"):
        tile_id = int(p.stem.split("_")[1])
        if (mask_dir / f"mask_{tile_id}.h5").exists():
            ids.append(tile_id)
    return sorted(ids)


def _split_ids(dataset_dir: Path) -> dict[Split, list[int]]:
    ids = _labeled_ids(dataset_dir)
    rng = random.Random(SPLIT_SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def _load_tile(dataset_dir: Path, tile_id: int) -> tuple[np.ndarray, np.ndarray]:
    img_path = dataset_dir / "TrainData" / "img" / f"image_{tile_id}.h5"
    mask_path = dataset_dir / "TrainData" / "mask" / f"mask_{tile_id}.h5"
    with h5py.File(img_path, "r") as f:
        img = f["img"][...].astype(np.float32)
    with h5py.File(mask_path, "r") as f:
        mask = f["mask"][...].astype(np.uint8)
    return img, mask


def load_split(
    split: Split,
    dataset_dir: Path,
    max_tiles: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load a fixed split as stacked arrays.

    Returns (X, y, ids): X is (N,128,128,14) float32, y is (N,128,128) uint8.
    max_tiles caps N for a cheap-proxy subset (e.g. to screen a round's candidate
    pipelines cheaply); the ids are always the first `max_tiles` of the full,
    seed-fixed split so the same subset is reused across agents and rounds.
    """
    ids = _split_ids(dataset_dir)[split]
    if max_tiles is not None:
        ids = ids[:max_tiles]

    imgs, masks = [], []
    for tile_id in ids:
        img, mask = _load_tile(dataset_dir, tile_id)
        imgs.append(img)
        masks.append(mask)

    X = np.stack(imgs, axis=0)
    y = np.stack(masks, axis=0)
    return X, y, ids


def build_data_cache(
    dataset_dir: Path,
    cache_path: Path,
    max_train_tiles: int,
    max_val_tiles: int,
) -> Path:
    """Build the shared train/val (+2-tile smoke sample) npz reused by every agent and
    round in a run, so the dataset is only ever loaded from disk once per run."""
    X_train, y_train, _ = load_split("train", dataset_dir, max_train_tiles)
    X_val, y_val, _ = load_split("val", dataset_dir, max_val_tiles)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        X_sample=X_train[:2],
        y_sample=y_train[:2],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    return cache_path
