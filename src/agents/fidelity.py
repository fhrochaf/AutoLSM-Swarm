"""Optional fidelity gate after call D: does the updated pipeline.py actually apply the
velocity it was given -- exactly that one change, correctly, and nothing else?

A judge LLM (the cheap role-2 model) reads the velocity, the diff from the previous
pipeline to the updated one, and the updated code. A NOT_FAITHFUL verdict sends the
update back to the code LLM (codegen.revise_pipeline_for_fidelity, optionally with Tavily
web search) to redo; the loop ends on a FAITHFUL verdict or after max_fidelity_iters
revisions. Switched on/off by config.fidelity_check. The prompts live in src/prompts.yaml
(group `fidelity`).

The gate only judges a velocity application. It does not judge round-0 pipelines (no
velocity yet) or runtime-crash fixes (that is the debug loop in agents/runner.py).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from agents.codegen import revise_pipeline_for_fidelity
from agents.prompts import DATASET_DESCRIPTION, REQUIRED_FUNCS_DOC, render
from agents.structured import FidelityVerdict, invoke_structured
from config import settings

# Last-resort parse of a plain-text reply: the "VERDICT: ..." line older prompts ask the judge to emit.
_VERDICT_RE = re.compile(r"(?im)^VERDICT:\s*(FAITHFUL|NOT[ _]FAITHFUL)\s*$")


@dataclass
class FidelityOutcome:
    code: str  # the pipeline to run: the first faithful one, else the last attempt
    faithful: bool  # False -> still flagged after max_fidelity_iters revisions
    checks: list[dict] = field(default_factory=list)  # one {"faithful": bool, "feedback": str} per judge call

    @property
    def revisions(self) -> int:
        "How many times the code LLM was asked to redo the update."
        return len(self.checks) - 1

    def report(self) -> str:
        "Human-readable log of every judge call (written to fidelity.md next to the pipeline)."
        parts = []
        for i, c in enumerate(self.checks, 1):
            parts.append(f"## Check {i}: {'FAITHFUL' if c['faithful'] else 'NOT_FAITHFUL'}\n\n{c['feedback'] or '(no findings)'}")
        end = (
            "Final pipeline passed the check." if self.faithful
            else "Still NOT_FAITHFUL after the last revision (config.run_unfaithful_pipeline decides whether "
                 "the latest attempt was run anyway or the update was rejected; the pipeline.py next to this "
                 "file is the one that ran)."
        )
        return "\n\n".join(parts + [end])


def unified_diff(old_code: str, new_code: str) -> str:
    diff = difflib.unified_diff(
        old_code.splitlines(), new_code.splitlines(), "previous pipeline.py", "updated pipeline.py", lineterm="", n=3
    )
    return "\n".join(diff) or "(no differences: the updated pipeline.py is identical to the previous one)"


def check_fidelity(
    judge_llm: BaseChatModel, velocity: str, old_code: str, new_code: str
) -> tuple[bool, str]:
    """Returns (faithful, feedback). The verdict is always requested as JSON (`violations`,
    `verdict`; agents/structured.py), whatever the prompt says. If no JSON can be obtained, a
    trailing "VERDICT: ..." line in the plain reply is still honored (what older prompts ask
    for); failing that it fails open (faithful=True) rather than blocking an update on a
    formatting hiccup."""
    parsed, raw = invoke_structured(
        judge_llm,
        FidelityVerdict,
        [
            (
                "system",
                render(
                    "fidelity.system",
                    required_funcs_doc=REQUIRED_FUNCS_DOC,
                    dataset_description=DATASET_DESCRIPTION,
                ),
            ),
            (
                "human",
                render(
                    "fidelity.user",
                    velocity=velocity,
                    diff=unified_diff(old_code, new_code),
                    new_code=new_code,
                ),
            ),
        ],
    )
    if parsed is not None:
        feedback = chr(10).join(f"- {v.strip()}" for v in parsed.violations if v.strip())
        return parsed.verdict == "FAITHFUL", feedback
    match = _VERDICT_RE.search(raw)
    if not match:
        return True, raw
    return match.group(1).upper() == "FAITHFUL", raw[: match.start()].strip()


def enforce_fidelity(
    judge_llm: BaseChatModel,
    code_llm: BaseChatModel,
    velocity: str,
    old_code: str,
    new_code: str,
    max_revisions: int,
    web_search: bool,
    reasoning: str = "",
) -> FidelityOutcome:
    """Judge `new_code` against `velocity`; while it is NOT_FAITHFUL, have the code LLM redo
    it (from `old_code`), up to `max_revisions` times. The judge sees only the velocity; the
    `reasoning` behind it goes to the code LLM alone, as context for the redo."""
    checks: list[dict] = []
    for attempt in range(max_revisions + 1):
        faithful, feedback = check_fidelity(judge_llm, velocity, old_code, new_code)
        checks.append({"faithful": faithful, "feedback": feedback})
        if faithful:
            return FidelityOutcome(new_code, True, checks)
        if attempt == max_revisions:
            break
        new_code = revise_pipeline_for_fidelity(
            code_llm, velocity, old_code, new_code, feedback, web_search, reasoning
        )
    return FidelityOutcome(new_code, False, checks)
