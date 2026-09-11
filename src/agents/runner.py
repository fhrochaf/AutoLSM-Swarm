"""The generate -> run -> debug-loop cycle: keeps regenerating pipeline.py from the
traceback until it runs end-to-end and passes the contract, or gives up after
max_debug_iters."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from agents import skill
from agents.codegen import fix_pipeline
from langchain_core.language_models import BaseChatModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
DRIVER_PATH = Path(__file__).parent / "driver.py"


class RunResult(BaseModel):
    success: bool
    dice: float | None = None
    iou: float | None = None
    debug_iters: int = 0
    final_code: str = ""
    error: str | None = None
    attempts: list[dict] = Field(default_factory=list)


def _run_once(pipeline_path: Path, data_npz_path: Path, timeout_s: int) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(DRIVER_PATH), str(SRC_DIR), str(pipeline_path), str(data_npz_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
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
    llm: BaseChatModel,
    data_npz_path: Path,
    max_debug_iters: int,
    timeout_s: int,
) -> RunResult:

    code = initial_code
    attempts: list[dict] = []

    for attempt in range(max_debug_iters + 1):
        pipeline_path = skill.write_pipeline(agent_dir, code)
        result = _run_once(pipeline_path, data_npz_path, timeout_s)
        attempts.append({"attempt": attempt, **result})

        if result.get("status") == "ok":
            (agent_dir / "driver_stdout.json").write_text(
                json.dumps(attempts, indent=2), encoding="utf-8"
            )
            return RunResult(
                success=True,
                dice=result.get("dice"),
                iou=result.get("iou"),
                debug_iters=attempt,
                final_code=code,
                attempts=attempts,
            )

        if attempt == max_debug_iters:
            break

        error_message = result.get("error", "unknown error")
        if result.get("traceback"):
            error_message = f"{error_message}\n{result['traceback']}"
        code = fix_pipeline(llm, code, error_message)

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
