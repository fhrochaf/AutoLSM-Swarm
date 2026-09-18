"""LLM-judge guardrail: does a generated pipeline.py actually implement the strategy
its skill.md describes, or did codegen silently drop/contradict it?
"""
from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from agents.prompts import DATASET_DESCRIPTION, REQUIRED_FUNCS_DOC
from config import settings

# Matches the trailing "VERDICT: ..." line the judge is asked to emit.
_VERDICT_RE = re.compile(r"(?im)^VERDICT:\s*(FAITHFUL|NOT FAITHFUL)\s*$")

FIDELITY_JUDGE_SYSTEM_PROMPT = f"""\
You are a code reviewer checking whether a generated pipeline.py actually implements \
the strategy described in its skill.md -- not whether it runs without errors, and not \
whether the strategy itself is good.

pipeline.py is ONLY required to implement\n: \
[{REQUIRED_FUNCS_DOC}]
\n
Dataset loading and evaluation-metric computation are handled by fixed harness code the agent never \
writes, even if skill.md mentions the dataset or these metrics as part of its \
methodology narrative. Never flag their absence from pipeline.py as a mismatch.

The pipeline only ever receives this fixed input:
{DATASET_DESCRIPTION}

If a skill.md claim requires data, channels, or metadata outside what this dataset \
provides (e.g. a sensor not among the listed input channels), the code cannot \
possibly honor it. Do not flag its absence as a mismatch, and do not require the code \
to invent or fake that data to satisfy skill.md.

Compare skill.md's other concrete, checkable claims (specific channel counts, named \
architectures/layers, named loss terms, specific training procedures such as \
cross-validation/TTA/LR schedulers, specific preprocessing steps, and so on) against what the code \
actually does. Ignore vague or aspirational language in skill.md that makes no concrete, \
checkable claim (e.g. "explore alternatives in future rounds" needs no corresponding \
code). Minor implementation details (variable names, an unpinned hyperparameter's \
exact value, code style) are not mismatches.

Before flagging a claim skill.md makes that the code doesn't literally do, check \
whether the code itself (a comment or docstring near the relevant logic) gives an \
explicit, concrete reason for doing something else instead -- e.g. the exact technique \
is infeasible given this dataset or the fixed pipeline contract, or it was deliberately \
replaced with a stated, reasonable substitute. Reserve NOT FAITHFUL for mechanisms skill.md \
states the pipeline uses that are silently absent from, or contradicted by, the code with no \
acknowledgment or reasoning at all.

Respond with a short bullet list of concrete mismatches found (empty if none), then end \
with exactly one line:
VERDICT: FAITHFUL
or
VERDICT: NOT FAITHFUL"""


def _build_prompt(skill_md: str, code: str) -> str:
    return f"""\
skill.md (the strategy this code is supposed to implement):
{skill_md}

pipeline.py (the code actually generated from it):
```python
{code}
```"""


def check_fidelity(judge_llm: BaseChatModel, skill_md: str, code: str) -> tuple[bool, str]:
    """Returns (faithful, feedback). Fails open (faithful=True) if the judge doesn't
    follow the VERDICT format, rather than blocking a round on a formatting hiccup."""
    response = judge_llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", FIDELITY_JUDGE_SYSTEM_PROMPT), ("human", _build_prompt(skill_md, code))]
    )
    text = response.text.strip()
    match = _VERDICT_RE.search(text)
    if not match:
        return True, text
    feedback = text[: match.start()].strip()
    return match.group(1).upper() == "FAITHFUL", feedback
