"""Standalone script executed as a subprocess to run + evaluate one agent's pipeline.py.

Runs in isolation from the orchestrator process (per the project's "local subprocess,
no sandbox" execution choice) so a crash, hang, or resource blowup in agent-generated
code cannot take down the main loop -- runner.py enforces a timeout around this.

Agents are allowed to import any library (deep learning frameworks included, not just
numpy/scikit-learn). If pipeline.py imports something that isn't installed yet, this
driver auto-installs it via `uv add --group agent_libraries <module>` -- a dependency
group (PEP 735) kept separate from the project's own base requirements -- then retries.
No allowlist: any missing top-level module name is attempted, up to a small bounded
number of distinct install attempts per run, so a genuinely bad/nonexistent package
name fails cleanly back into the LLM debug loop instead of looping forever.

Invoked as: python driver.py <src_dir> <pipeline_path> <data_npz_path>
Prints exactly one JSON line to stdout:
  {"status": "ok", "dice": <float>, "iou": <float>}
  {"status": "error", "error": "<message>", "traceback": "<traceback or "">"}
Progress/diagnostics (e.g. package installs) go to stderr, never stdout, so they never
interfere with that single JSON line.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

MAX_INSTALL_ATTEMPTS = 5
INSTALL_TIMEOUT_S = 900
AGENT_LIBRARIES_GROUP = "agent_libraries"


def _install_package(module_name: str, repo_root: str) -> tuple[bool, str]:
    """Best-effort install of a missing import via uv, recorded in a dependency group
    that's excluded from the project's own base requirements."""
    try:
        proc = subprocess.run(
            ["uv", "add", "--group", AGENT_LIBRARIES_GROUP, module_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, f"install of {module_name!r} timed out after {INSTALL_TIMEOUT_S}s"
    if proc.returncode != 0:
        return False, f"'uv add --group {AGENT_LIBRARIES_GROUP} {module_name}' failed:\n{proc.stderr[-2000:]}"
    importlib.invalidate_caches()
    return True, ""


def _run_pipeline_once(pipeline_path: str, data_npz_path: str) -> dict:
    """One full attempt: (re)load pipeline.py fresh and run it end-to-end. Raises
    ModuleNotFoundError if the module (or something it lazily imports) isn't installed."""
    import numpy as np

    from agents.pipeline_contract import smoke_test, validate_module
    from eval.infer import evaluate

    spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    validate_module(mod)

    data = np.load(data_npz_path)
    smoke_test(mod, data["X_sample"], data["y_sample"])

    X_processed = mod.preprocess(data["X_train"])
    model = mod.build_model(None)
    model = mod.train(model, X_processed, data["y_train"])

    return evaluate(mod, model, data["X_val"], data["y_val"])


def main() -> None:
    src_dir, pipeline_path, data_npz_path = sys.argv[1:4]
    sys.path.insert(0, src_dir)
    repo_root = str(Path(src_dir).parent)

    from agents.pipeline_contract import ContractError

    attempted: set[str] = set()

    try:
        while True:
            try:
                scores = _run_pipeline_once(pipeline_path, data_npz_path)
                print(json.dumps({"status": "ok", **scores}))
                return
            except ModuleNotFoundError as exc:
                missing = (exc.name or "").split(".")[0]
                if not missing or missing in attempted or len(attempted) >= MAX_INSTALL_ATTEMPTS:
                    raise
                attempted.add(missing)
                print(f"[driver] missing module {missing!r}; installing...", file=sys.stderr)
                ok, err = _install_package(missing, repo_root)
                if not ok:
                    raise ModuleNotFoundError(f"could not auto-install {missing!r}: {err}") from exc
                print(f"[driver] installed {missing!r}; retrying pipeline...", file=sys.stderr)
    except ContractError as exc:
        print(json.dumps({"status": "error", "error": str(exc), "traceback": ""}))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - report any failure back to the debug loop
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
