"""LLM calls that produce a skill.md or a pipeline.py. The only place the LLM is invoked
for the agent's own generate/debug cycle (see orchestration/peer_review.py for the
Reflect/VelocityUpdate/SkillUpdate calls that revise skill_md between rounds)."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch

from agents.prompts import DATASET_DESCRIPTION, PIPELINE_SYSTEM_PROMPT, SKILL_SYSTEM_PROMPT
from agents.text_utils import extract_code, strip_optional_fence

# Cap on search-then-respond round trips inside revise_pipeline_for_fidelity, so an
# already-expensive debug loop can't be blown up further by an open-ended tool loop.
_MAX_SEARCH_ITERS = 3


def write_skill(llm: BaseChatModel, retrieved_papers: str) -> str:
    prompt = f"""\
{DATASET_DESCRIPTION}

Retrieved literature (paper id, methods, datasets, novelty):
{retrieved_papers}

Write your initial skill.md. Keep it concrete and actionable: state the specific
preprocessing steps, the specific model/architecture choice, and cite which paper id(s)
each choice comes from."""
    response = llm.invoke([("system", SKILL_SYSTEM_PROMPT), ("human", prompt)])
    # .text handles both plain-string content and content-block lists (e.g. a
    # reasoning model's text+thinking blocks), regardless of provider.
    return strip_optional_fence(response.text)


def generate_pipeline(llm: BaseChatModel, skill_md: str) -> str:
    prompt = f"""\
{DATASET_DESCRIPTION}

Your current skill.md (your strategy for this round):
{skill_md}

Write pipeline.py implementing this strategy under the fixed contract."""
    response = llm.invoke([("system", PIPELINE_SYSTEM_PROMPT), ("human", prompt)])
    return extract_code(response.text)


def fix_pipeline(llm: BaseChatModel, previous_code: str, error_message: str) -> str:
    prompt = f"""\
The following pipeline.py failed when executed:

```python
{previous_code}
```

Error:
{error_message}

Fix the module so it satisfies the contract and runs end-to-end without errors. Output \
ONLY the corrected Python code for pipeline.py, in a single ```python code block."""
    response = llm.invoke([("system", PIPELINE_SYSTEM_PROMPT), ("human", prompt)])
    return extract_code(response.text)


def revise_pipeline_for_fidelity(
    llm: BaseChatModel, skill_md: str, previous_code: str, feedback: str
) -> str:
    """Like fix_pipeline, but for a judge-flagged skill<->code mismatch rather than a
    runtime error. Given a search tool: some mismatches are "you got the technique
    wrong" (e.g. skill.md names Dempster-Shafer fusion, the code has a lookalike
    weighted average) rather than "you omitted it entirely" -- letting the model look up
    the actual formula/algorithm for a named technique it's unsure of, instead of
    guessing from parametric memory, targets that specific failure mode."""
    search = TavilySearch(max_results=3)
    llm_with_tools = llm.bind_tools([search])

    prompt = f"""\
Your current skill.md (the strategy this code is supposed to implement):
{skill_md}

The following pipeline.py was generated from it:

```python
{previous_code}
```

A reviewer found concrete mismatches between this code and skill.md:
{feedback}

If you are unsure of the exact formula/algorithm behind a named technique skill.md \
mentions, use the search tool to look it up rather than guessing. Rewrite pipeline.py \
so it satisfies the contract and faithfully implements skill.md. Output ONLY the \
corrected Python code for pipeline.py, in a single ```python code block."""

    messages: list = [SystemMessage(PIPELINE_SYSTEM_PROMPT), HumanMessage(prompt)]
    for _ in range(_MAX_SEARCH_ITERS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return extract_code(response.text)
        for call in response.tool_calls:
            result = search.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Still calling tools after _MAX_SEARCH_ITERS -- cut it off and force a plain answer.
    final = llm.invoke(messages)
    return extract_code(final.text)
