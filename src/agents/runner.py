"""The run -> debug-loop cycle: keeps fixing pipeline.py from the traceback until it runs
end-to-end under every evaluation seed and passes the contract, or gives up after
max_debug_iters.

One "run" is one driver.py subprocess per seed (AUTOLSM_SEED=<seed>), each training the
pipeline from scratch. The score is the mean over seeds; a crash on any seed sends the
whole pipeline back through the debug loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agents import artifacts
from agents.codegen import fix_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
DRIVER_PATH = Path(__file__).parent / "driver.py"


class RunResult(BaseModel):
    success: bool
    dice: float | None = None  # mean over seeds -- the score selection is based on
    iou: float | None = None  # mean over seeds
    dice_per_seed: dict[int, float] = Field(default_factory=dict)
    dice_std: float | None = None
    hparams: dict = Field(default_factory=dict)
    debug_iters: int = 0
    final_code: str = ""
    error: str | None = None
    attempts: list[dict] = Field(default_factory=list)


def _run_once(pipeline_path: Path, data_npz_path: Path, timeout_s: int, seed: int) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(DRIVER_PATH), str(SRC_DIR), str(pipeline_path), str(data_npz_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "AUTOLSM_SEED": str(seed)},
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"timed out after {timeout_s}s", "traceback": ""}

    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "driver produced no parseable result",
            "traceback": f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        }


def run_agent_round(
    agent_dir: Path,
    initial_code: str,
    code_llm: BaseChatModel,
    data_npz_path: Path,
    max_debug_iters: int,
    timeout_s: int,
    seeds: list[int],
) -> RunResult:
    code = initial_code
    attempts: list[dict] = []

    for attempt in range(max_debug_iters + 1):
        pipeline_path = artifacts.write_pipeline(agent_dir, code)

        ok_runs: list[dict] = []
        failure: dict | None = None
        for seed in seeds:
            result = _run_once(pipeline_path, data_npz_path, timeout_s, seed)
            attempts.append({"attempt": attempt, "seed": seed, **result})
            if result.get("status") != "ok":
                failure = {**result, "error": f"[seed {seed}] {result.get('error', 'unknown error')}"}
                break
            ok_runs.append(result)

        if failure is None:
            (agent_dir / "driver_stdout.json").write_text(
                json.dumps(attempts, indent=2), encoding="utf-8"
            )
            dices = [r["dice"] for r in ok_runs]
            return RunResult(
                success=True,
                dice=float(np.mean(dices)),
                iou=float(np.mean([r["iou"] for r in ok_runs])),
                dice_per_seed={r["seed"]: r["dice"] for r in ok_runs},
                dice_std=float(np.std(dices)),
                hparams=ok_runs[0].get("hparams", {}),
                debug_iters=attempt,
                final_code=code,
                attempts=attempts,
            )

        if attempt == max_debug_iters:
            break

        error_message = failure["error"]
        if failure.get("traceback"):
            error_message = f"{error_message}\n{failure['traceback']}"
        code = fix_pipeline(code_llm, code, error_message)

    (agent_dir / "driver_stdout.json").write_text(
        json.dumps(attempts, indent=2), encoding="utf-8"
    )
    return RunResult(
        success=False,
        debug_iters=max_debug_iters,
        final_code=code,
        error=attempts[-1].get("error"),
        attempts=attempts,
    )
