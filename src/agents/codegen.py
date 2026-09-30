"""LLM calls that produce or change a pipeline.py -- the only artifact an agent owns.

Three entry points, all run on the "code" model: write the very first pipeline from
retrieved literature (round 0), apply one velocity to the current pipeline (call D of
the peer-review chain in orchestration/peer_review.py), and fix a pipeline that crashed
(the debug loop in agents/runner.py). Their prompts live in src/prompts.yaml.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agents.prompts import DATASET_DESCRIPTION, PIPELINE_SYSTEM_PROMPT, render
from agents.text_utils import extract_code
from config import settings


def _invoke_code(llm: BaseChatModel, prompt: str) -> str:
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", PIPELINE_SYSTEM_PROMPT), ("human", prompt)]
    )
    # .text handles both plain-string content and content-block lists (e.g. a
    # reasoning model's text+thinking blocks), regardless of provider.
    return extract_code(response.text)


def generate_initial_pipeline(llm: BaseChatModel, retrieved_papers: str) -> str:
    """Round 0: a first pipeline.py chosen straight from the retrieved literature."""
    return _invoke_code(
        llm,
        render(
            "pipeline.initial_user",
            dataset_description=DATASET_DESCRIPTION,
            retrieved_papers=retrieved_papers,
        ),
    )


def apply_velocity(llm: BaseChatModel, current_code: str, velocity: str) -> str:
    """Call D: turn the current pipeline.py plus one velocity into the next pipeline.py.

    The velocity names exactly one change; everything else must stay as it is, so that
    the next round's score change can be attributed to that one change."""
    return _invoke_code(
        llm,
        render(
            "pipeline.apply_velocity_user",
            dataset_description=DATASET_DESCRIPTION,
            current_code=current_code,
            velocity=velocity,
        ),
    )


def fix_pipeline(llm: BaseChatModel, previous_code: str, error_message: str) -> str:
    return _invoke_code(
        llm,
        render("pipeline.fix_user", previous_code=previous_code, error_message=error_message),
    )
