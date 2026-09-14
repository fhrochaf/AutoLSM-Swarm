"""LLM-judge guardrail: does a generated pipeline.py actually implement the strategy
its skill.md describes, or did codegen silently drop/contradict it?
"""
from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from agents.pipeline_contract import REQUIRED_FUNCS_DOC

# Matches the trailing "VERDICT: ..." line the judge is asked to emit.
_VERDICT_RE = re.compile(r"(?im)^VERDICT:\s*(FAITHFUL|NOT FAITHFUL)\s*$")

FIDELITY_JUDGE_SYSTEM_PROMPT = f"""\
You are a strict code reviewer checking whether a generated pipeline.py actually \
implements the strategy described in its skill.md -- not whether it runs without \
errors, and not whether the strategy itself is good.

pipeline.py is ONLY required to implement\n: \
[{REQUIRED_FUNCS_DOC}]
\n
Dataset loading and evaluation-metric computation are handled by fixed harness code the agent never \
writes, even if skill.md mentions the dataset or these metrics as part of its \
methodology narrative. Never flag their absence from pipeline.py as a mismatch.

Compare skill.md's other concrete, checkable claims (specific channel counts, named \
architectures/layers, named loss terms, specific training procedures such as \
cross-validation/TTA/LR schedulers, specific preprocessing steps, and so on) against what the code \
actually does. Ignore vague or aspirational language in skill.md that makes no concrete, \
checkable claim (e.g. "explore alternatives in future rounds" needs no corresponding \
code). Flag only clear, concrete mismatches: a mechanism skill.md states the pipeline \
uses that is absent from, or contradicted by, the code. Minor implementation details \
(variable names, an unpinned hyperparameter's exact value, code style) are not \
mismatches.

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
    response = judge_llm.invoke(
        [("system", FIDELITY_JUDGE_SYSTEM_PROMPT), ("human", _build_prompt(skill_md, code))]
    )
    text = response.text.strip()
    match = _VERDICT_RE.search(text)
    if not match:
        return True, text
    feedback = text[: match.start()].strip()
    return match.group(1).upper() == "FAITHFUL", feedback
