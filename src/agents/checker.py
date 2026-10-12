"""The HYPOTHESIS CHECKER: answers one agent's check request during the conversation.

An agent hands off to it (tool request_hypothesis_check -> graph node hypothesis_checker) with
a question. The checker is a create_agent() agent (role-3 LLM) with one tool, run_analysis,
which executes a Python script in a subprocess (agents/checker_driver.py) over this round's
validation predictions. It reads the output, iterates, and its final answer is posted on the
blackboard.

What the checker can see is deliberately narrow: every agent's predicted validation masks
(first evaluation seed) and the validation ground-truth masks needed to score them. No
images, no train split, and never the test split.
"""
from __future__ import annotations

import subprocess
import sys
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
from orchestration.state import PendingCheck, SwarmState

CHECKER_DRIVER = Path(__file__).parent / "checker_driver.py"


def checker_dir(state: SwarmState, settings: Settings) -> Path:
    d = artifacts.round_dir(settings.runs_dir, state.run_id, state.round) / "checker"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_inputs(state: SwarmState, settings: Settings) -> Path:
    """round_<t>/checker/inputs.npz: y_val plus pred_agent_<i> for every agent whose pipeline
    ran this round. Written once per round."""
    path = checker_dir(state, settings) / "inputs.npz"
    if path.exists():
        return path
    seed = settings.active_seeds()[0]
    arrays = {"y_val": np.load(state.data_npz_path)["y_val"].astype(bool)}
    for a in state.agents:
        preds_path = artifacts.val_preds_path(
            artifacts.agent_round_dir(settings.runs_dir, state.run_id, state.round, a.agent_idx), seed
        )
        if a.last_error is None and preds_path.exists():
            arrays[f"pred_agent_{a.agent_idx}"] = np.load(preds_path)["preds"].astype(bool)
    np.savez_compressed(path, **arrays)
    return path


def data_schema(inputs_path: Path) -> str:
    with np.load(inputs_path) as npz:
        lines = [f"- DATA[{k!r}]: {npz[k].dtype} array of shape {npz[k].shape}" for k in npz.files]
    return "\n".join(lines) + (
        "\nAll arrays are binary masks of the same validation tiles, in the same order (axis 0 = tile; "
        "True = landslide). y_val is the ground truth; pred_agent_<i> is agent_<i>'s prediction. "
        "Agents whose pipeline failed this round have no pred_agent_<i>."
    )


def make_run_analysis_tool(inputs_path: Path, workdir: Path, check_idx: int, settings: Settings) -> BaseTool:
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
                [sys.executable, str(CHECKER_DRIVER), str(script), str(inputs_path)],
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
    tool_ = make_run_analysis_tool(inputs, workdir, check_idx, settings)
    system = render("checker.system", data_schema=data_schema(inputs), max_runs=settings.checker_max_runs)
    user = render("checker.user", requester=pending.requester, question=pending.question, scores=scores_table(state))
    out = invoke_agent(
        make_llm(settings, 3), [tool_], system, [HumanMessage(user)], settings,
        name="hypothesis_checker", max_tool_calls=settings.checker_max_runs,
    )
    answer = final_text(out) or "(the checker produced no answer)"
    log = f"# Check {check_idx} (requested by agent_{pending.requester})\n\n## Question\n{pending.question}\n\n{transcript(out, system)}"
    (workdir / f"check_{check_idx}.md").write_text(log, encoding="utf-8")
    return answer
