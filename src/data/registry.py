"""Name -> DatasetSpec lookup. Register a new dataset here after adding its module
under data/datasets/ -- this is the only place that needs to know every dataset exists.
"""
from __future__ import annotations

from data.datasets import landslide4sense
from data.spec import DatasetSpec

_REGISTRY: dict[str, DatasetSpec] = {
    landslide4sense.SPEC.name: landslide4sense.SPEC,
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Registered datasets: {sorted(_REGISTRY)}"
        ) from None
