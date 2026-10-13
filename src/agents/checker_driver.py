"""Subprocess entry point for one HYPOTHESIS CHECKER analysis script (agents/checker.py).

    python checker_driver.py <script.py> <inputs.npz> <data_cache.npz>

Executes the script with np and a read-only mapping DATA in scope; whatever the script prints is
the result. DATA holds every array of inputs.npz (masks, tile groups) plus 'X_val', the validation
images, read from the run's data cache only when a script first uses it (it is large). Run as a
subprocess (like driver.py) so a crashing or runaway script cannot take down the orchestrator.
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np

LAZY_FROM_CACHE = ("X_val",)  # never train/test: the checker only ever sees validation tiles


class Data(Mapping):
    def __init__(self, inputs_path: Path, cache_path: Path):
        with np.load(inputs_path) as npz:
            self._arrays = {k: npz[k] for k in npz.files}
        for arr in self._arrays.values():
            arr.setflags(write=False)
        self._cache_path = cache_path

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._arrays and key in LAZY_FROM_CACHE:
            with np.load(self._cache_path) as npz:
                arr = npz[key]
            arr.setflags(write=False)
            self._arrays[key] = arr
        return self._arrays[key]

    def __iter__(self):
        return iter([*self._arrays, *(k for k in LAZY_FROM_CACHE if k not in self._arrays)])

    def __len__(self) -> int:
        return len(set(self._arrays) | set(LAZY_FROM_CACHE))


def main() -> int:
    script_path, inputs_path, cache_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    data = Data(inputs_path, cache_path)
    try:
        code = compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
        exec(code, {"__name__": "__main__", "DATA": data, "np": np})
    except Exception:  # noqa: BLE001 - the traceback is the result the checker reads
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
