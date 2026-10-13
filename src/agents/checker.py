"""The HYPOTHESIS CHECKER: answers one agent's check request during the conversation.

An agent hands off to it (tool request_hypothesis_check -> graph node hypothesis_checker) with
a question. The checker is a create_agent() agent (role-3 LLM) with one tool, run_analysis,
which executes a Python script in a subprocess (agents/checker_driver.py) over this round's
validation predictions. It reads the output, iterates, and its final answer is posted on the
blackboard.

What the checker can see is the validation split only: every agent's predicted masks (first
evaluation seed), the ground-truth masks, the observer's difficulty group of each tile, and the
validation images (X_val, read lazily from the run's data cache). Never the train or test split.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool

from agents import artifacts
from agents.context import scores_table
from agents.prompts import render
from agents.swarm_agent import final_text, invoke_agent, make_llm, transcript
from agents.text_utils import cap_text
from config import Settings
from eval import observe_detail
from orchestration.state import PendingCheck, SwarmState

CHECKER_DRIVER = Path(__file__).parent / "checker_driver.py"


def checker_dir(state: SwarmState, settings: Settings) -> Path:
    d = artifacts.round_dir(settings.runs_dir, state.run_id, state.round) / "checker"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tile_groups(state: SwarmState, settings: Settings, y_val: np.ndarray) -> np.ndarray | None:
    """The observer's per-tile difficulty group (eval/observe_detail.py), over every pipeline of the
    run so far -- so "persistently hard tiles" means the same to the checker as to the agents."""
    seed = settings.active_seeds()[0]
    entries = []
    for r in range(state.round + 1):
        for a in state.agents:
            d = artifacts.agent_round_dir(settings.runs_dir, state.run_id, r, a.agent_idx)
            entries.append((r, a.agent_idx, artifacts.val_preds_path(d, seed), d / "tile_stats.npz"))
    gt_px = y_val.reshape(len(y_val), -1).sum(axis=1)
    history = observe_detail._history(entries, y_val, gt_px)
    if history is None:
        return None
    cat_pos, cat_neg = observe_detail._classify(history, gt_px)
    pos_names = np.array(["solved", "contested", "persistently_hard"])
    neg_names = np.array(["clean", "intermittent_false_alarm", "persistent_false_alarm"])
    return np.where(gt_px > 0, pos_names[np.clip(cat_pos, 0, 2)], neg_names[np.clip(cat_neg, 0, 2)])


def prepare_inputs(state: SwarmState, settings: Settings) -> Path:
    """round_<t>/checker/inputs.npz: y_val, pred_agent_<i> for every agent whose pipeline ran this
    round, and the observer's tile_group per tile. Written once per round. The validation images
    are not copied: the driver reads X_val from the run's data cache on first use."""
    path = checker_dir(state, settings) / "inputs.npz"
    if path.exists():
        return path
    seed = settings.active_seeds()[0]
    y_val = np.load(state.data_npz_path)["y_val"].astype(bool)
    arrays = {"y_val": y_val}
    for a in state.agents:
        preds_path = artifacts.val_preds_path(
            artifacts.agent_round_dir(settings.runs_dir, state.run_id, state.round, a.agent_idx), seed
        )
        if a.last_error is None and preds_path.exists():
            arrays[f"pred_agent_{a.agent_idx}"] = np.load(preds_path)["preds"].astype(bool)
    groups = _tile_groups(state, settings, y_val)
    if groups is not None:
        arrays["tile_group"] = groups
    np.savez_compressed(path, **arrays)
    return path


def _npz_member_shape(npz_path: Path, name: str) -> tuple[tuple, str]:
    """Shape and dtype of one array in an .npz, read from its header without loading it."""
    with zipfile.ZipFile(npz_path) as zf, zf.open(f"{name}.npy") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, _, dtype = read_header(f)
    return shape, str(dtype)


def data_schema(inputs_path: Path, data_npz_path: Path) -> str:
    with np.load(inputs_path) as npz:
        lines = [f"- DATA[{k!r}]: {npz[k].dtype} {npz[k].shape}" for k in npz.files]
    shape, dtype = _npz_member_shape(data_npz_path, "X_val")
    lines.append(f"- DATA['X_val']: {dtype} {shape} -- the validation images, loaded on first access (slow, ~{np.prod(shape) * np.dtype(dtype).itemsize / 2**30:.1f} GB; use only if needed)")
    return "\n".join(lines) + (
        "\nAxis 0 = the same validation tiles everywhere. Masks: True = landslide; y_val = ground truth, "
        "pred_agent_<i> = agent_<i>'s prediction (absent if its pipeline failed). "
        "X_val channels: 0-11 = Sentinel-2 B1-B12 (red B4 = 3, NIR B8 = 7, so NDVI = (X[...,7]-X[...,3])/(X[...,7]+X[...,3])), "
        "12 = slope, 13 = DEM. tile_group = the observer's difficulty group of each tile, as named in its report."
    )


def make_run_analysis_tool(
    inputs_path: Path, data_npz_path: Path, workdir: Path, check_idx: int, settings: Settings
) -> BaseTool:
    runs = {"n": 0}

    @tool
    def run_analysis(code: str) -> str:
        """Execute a Python analysis script over DATA (see the system prompt) and return what it
        prints (stdout, then stderr if any)."""
        runs["n"] += 1
        script = workdir / f"check_{check_idx}_{runs['n']}.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(CHECKER_DRIVER), str(script), str(inputs_path), str(data_npz_path)],
                capture_output=True, text=True, timeout=settings.checker_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {settings.checker_timeout_s}s -- write a cheaper script"
        out = proc.stdout.strip() or "(nothing printed)"
        if proc.stderr.strip():
            out += f"\n\nstderr:\n{proc.stderr.strip()}"
        return cap_text(out, settings.checker_output_char_cap)

    return run_analysis


def run_check(state: SwarmState, pending: PendingCheck, settings: Settings) -> str:
    """Answer one check request; returns the post for the blackboard (the full transcript is
    saved to round_<t>/checker/check_<k>.md)."""
    inputs = prepare_inputs(state, settings)
    workdir = checker_dir(state, settings)
    check_idx = state.checks_used  # already incremented for this request by the talk node
    data_npz = Path(state.data_npz_path)
    tool_ = make_run_analysis_tool(inputs, data_npz, workdir, check_idx, settings)
    system = render("checker.system", data_schema=data_schema(inputs, data_npz), max_runs=settings.checker_max_runs)
    user = render("checker.user", requester=pending.requester, question=pending.question, scores=scores_table(state))
    out = invoke_agent(
        make_llm(settings, 3), [tool_], system, [HumanMessage(user)], settings,
        name="hypothesis_checker", max_tool_calls=settings.checker_max_runs,
    )
    answer = final_text(out) or "(the checker produced no answer)"
    log = f"# Check {check_idx} (requested by agent_{pending.requester})\n\n## Question\n{pending.question}\n\n{transcript(out, system)}"
    (workdir / f"check_{check_idx}.md").write_text(log, encoding="utf-8")
    return answer
