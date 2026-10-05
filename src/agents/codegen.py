"""LLM calls that produce or change a pipeline.py -- the only artifact an agent owns.

Four entry points, all run on the "code" model: write the very first pipeline from
retrieved literature (round 0), apply one velocity to the current pipeline (call D of
the peer-review chain in orchestration/peer_review.py), redo an update the fidelity judge
flagged as NOT_FAITHFUL (agents/fidelity.py), and fix a pipeline that crashed (the debug
loop in agents/runner.py). Their prompts live in src/prompts.yaml.
"""
from __future__ import annotations

import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.prompts import DATASET_DESCRIPTION, PIPELINE_SYSTEM_PROMPT, render
from agents.text_utils import extract_code
from config import settings

# Cap on search-then-respond round trips inside revise_pipeline_for_fidelity, so an
# already-expensive revision can't be blown up further by an open-ended tool loop.
_MAX_SEARCH_ITERS = 3
_SEARCH_HINT = (
    "If you are unsure of the exact formula or algorithm behind a named technique, use the "
    "search tool to look it up rather than guessing from memory. "
)


def _invoke_code(llm: BaseChatModel, prompt: str) -> str:
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", PIPELINE_SYSTEM_PROMPT), ("human", prompt)]
    )
    # .text handles both plain-string content and content-block lists (e.g. a
    # reasoning model's text+thinking blocks), regardless of provider.
    return extract_code(response.text)


def generate_initial_pipeline(llm: BaseChatModel, retrieved_papers: str | None) -> str:
    """Round 0: a first pipeline.py chosen straight from the retrieved literature, or, when
    retrieved_papers is None (settings.use_rag off), from the dataset description alone."""
    if retrieved_papers is None:
        return _invoke_code(
            llm, render("pipeline.initial_user_no_rag", dataset_description=DATASET_DESCRIPTION)
        )
    return _invoke_code(
        llm,
        render(
            "pipeline.initial_user",
            dataset_description=DATASET_DESCRIPTION,
            retrieved_papers=retrieved_papers,
        ),
    )


def apply_velocity(llm: BaseChatModel, current_code: str, velocity: str, reasoning: str = "") -> str:
    """Call D: turn the current pipeline.py plus one velocity into the next pipeline.py.

    The velocity names the change(s) to make -- how many is up to the prompts in use (one per
    round in prompts.yaml, possibly several in a multi-change variant); this function does not
    care. Everything the velocity does not name must stay as it is. `reasoning` is the thinking
    behind the velocity; the code LLM gets it as context (it is NOT what it applies)."""
    return _invoke_code(
        llm,
        render(
            "pipeline.apply_velocity_user",
            dataset_description=DATASET_DESCRIPTION,
            current_code=current_code,
            velocity=velocity,
            velocity_reasoning=reasoning.strip() or "(none given)",
        ),
    )


def fix_pipeline(llm: BaseChatModel, previous_code: str, error_message: str) -> str:
    return _invoke_code(
        llm,
        render("pipeline.fix_user", previous_code=previous_code, error_message=error_message),
    )


def _make_search_tool():
    """Tavily web search for the fidelity revision, or None (with a note on stderr) if it
    can't be built -- missing langchain-tavily or TAVILY_API_KEY -- so the gate degrades to a
    plain revision instead of crashing the round."""
    try:
        from langchain_tavily import TavilySearch

        return TavilySearch(max_results=3)
    except Exception as exc:  # noqa: BLE001 - any setup failure just disables the tool
        print(
            f"[fidelity] web search unavailable ({type(exc).__name__}; is TAVILY_API_KEY set in .env?) "
            "-- revising without it",
            file=sys.stderr,
        )
        return None


def revise_pipeline_for_fidelity(
    llm: BaseChatModel,
    velocity: str,
    previous_code: str,
    flawed_code: str,
    feedback: str,
    web_search: bool = False,
    reasoning: str = "",
) -> str:
    """Redo call D after the judge flagged the result as NOT_FAITHFUL. Starts again from the
    pipeline BEFORE the velocity, shown together with the flawed attempt and the reviewer's
    findings. With web_search, the model may look up the exact formula/algorithm of a named
    technique (some mismatches are "you used a lookalike", not "you left it out") instead of
    guessing from memory."""
    search = _make_search_tool() if web_search else None
    prompt = render(
        "fidelity.revise_user",
        dataset_description=DATASET_DESCRIPTION,
        velocity=velocity,
        velocity_reasoning=reasoning.strip() or "(none given)",
        previous_code=previous_code,
        flawed_code=flawed_code,
        feedback=feedback or "(the reviewer gave no details)",
        search_hint=_SEARCH_HINT if search is not None else "",
    )
    if search is None:
        return _invoke_code(llm, prompt)

    llm_with_tools = llm.bind_tools([search]).with_retry(stop_after_attempt=settings.llm_retry_attempts)
    messages: list = [SystemMessage(PIPELINE_SYSTEM_PROMPT), HumanMessage(prompt)]
    for _ in range(_MAX_SEARCH_ITERS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return extract_code(response.text)
        for call in response.tool_calls:
            try:
                result = str(search.invoke(call["args"]))
            except Exception as exc:  # noqa: BLE001 - a failed search must not sink the round
                result = f"search failed: {exc!r}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # Still calling tools after _MAX_SEARCH_ITERS -- cut it off and force a plain answer.
    final = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(messages)
    return extract_code(final.text)
