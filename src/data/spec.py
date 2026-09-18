"""The declarative description of one dataset: what it looks like (for LLM prompts and
the pipeline contract's doc-string) and how to load it (for the actual training code).

Adding a new dataset means writing one module under data/datasets/ that builds a
DatasetSpec and registering it in data/registry.py -- nothing elsewhere (prompts,
pipeline_contract, dataset.py) should need to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

Split = Literal["train", "val", "test"]

# (split, dataset_dir, max_tiles) -> (X, y, ids)
LoadSplitFn = Callable[[Split, Path, "int | None"], tuple[np.ndarray, np.ndarray, list[int]]]
# (dataset_dir, tile_ids) -> (X, y)
LoadSmokeSampleFn = Callable[[Path, list[int]], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything dataset-specific that the rest of the codebase needs.

    description: free-text prose handed to the LLM (skill/pipeline prompts, the
        retrieval query, the fidelity judge) -- shape, channels, temporal structure,
        label semantics, whatever an agent needs to know about the fixed input it
        will receive.
    input_shape_doc / label_shape_doc: short shape annotations (e.g. "(N,128,128,14)",
        "(N,128,128)") spliced into the pipeline contract's doc-string
        (agents/pipeline_contract.py's build_required_funcs_doc). Keep these
        consistent with what load_split/load_smoke_sample actually return -- they are
        documentation for the LLM, not enforced by validate_module/smoke_test, which
        stay shape-agnostic on purpose.
    smoke_tile_ids: a handful of ids from the train split, hand-verified to include at
        least one tile with a positive label and one without, used by
        pipeline_contract.py's smoke_test. Must be re-verified for each new dataset.
    load_split / load_smoke_sample: the actual per-dataset loading logic.
    """

    name: str
    description: str
    input_shape_doc: str
    label_shape_doc: str
    smoke_tile_ids: list[int]
    load_split: LoadSplitFn
    load_smoke_sample: LoadSmokeSampleFn
