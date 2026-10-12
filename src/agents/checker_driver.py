"""Subprocess entry point for one HYPOTHESIS CHECKER analysis script (agents/checker.py).

    python checker_driver.py <script.py> <inputs.npz>

Loads inputs.npz into a read-only dict DATA and executes the script with DATA and np in
scope; whatever the script prints is the result. Run as a subprocess (like driver.py) so a
crashing or runaway script cannot take down the orchestrator.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np


def main() -> int:
    script_path, inputs_path = Path(sys.argv[1]), Path(sys.argv[2])
    with np.load(inputs_path) as npz:
        data = {k: npz[k] for k in npz.files}
    for arr in data.values():
        arr.setflags(write=False)
    try:
        code = compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
        exec(code, {"__name__": "__main__", "DATA": data, "np": np})
    except Exception:  # noqa: BLE001 - the traceback is the result the checker reads
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
