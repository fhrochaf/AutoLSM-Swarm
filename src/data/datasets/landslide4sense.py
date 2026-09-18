"""Landslide4Sense: fixed train/val/test split + loading for the labeled tiles.
We carve our own train/val/test split out of the labeled tiles, once, with a fixed seed.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from config import settings
from data.dataset import split_ids_by_ratio
from data.spec import DatasetSpec, Split

DESCRIPTION = """\
Dataset Descriptipon:
[Each tile is 128x128 pixels with 14 input channels that include:
- Multispectral data from Sentinel-2: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.
- Slope data from ALOS PALSAR: B13.
- Digital elevation model (DEM) from ALOS PALSAR: B14.
- Only post-event images, which may or may not contain landslides (no temporal series).
The label is a binary landslide/no-landslide mask, 128x128, values in {0,1}.
X arrays are shaped (N,128,128,14) float32; y arrays are shaped (N,128,128) uint8.]
"""


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
    return split_ids_by_ratio(ids, settings.seed, settings.split_ratios)


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


def load_smoke_sample(dataset_dir: Path, tile_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Load pipeline_contract.py's fixed smoke-test tiles (SPEC.smoke_tile_ids) by id."""
    imgs, masks = [], []
    for tile_id in tile_ids:
        img, mask = _load_tile(dataset_dir, tile_id)
        imgs.append(img)
        masks.append(mask)
    return np.stack(imgs, axis=0), np.stack(masks, axis=0)


# Verified once against the dataset with seed=42: tiles 1, 4 have at least one
# landslide pixel; 8, 10 have none. Re-verify when changing the seed or split ratios.
SPEC = DatasetSpec(
    name="landslide4sense",
    description=DESCRIPTION,
    input_shape_doc="(N,128,128,14)",
    label_shape_doc="(N,128,128)",
    smoke_tile_ids=[1, 4, 8, 10],
    load_split=load_split,
    load_smoke_sample=load_smoke_sample,
)
