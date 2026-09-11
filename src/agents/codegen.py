"""LLM calls that produce a skill.md or a pipeline.py. The only place the LLM is invoked
for the agent's own generate/debug cycle (see orchestration/peer_review.py for the
Reflect/VelocityUpdate/SkillUpdate calls that revise skill_md between rounds)."""
from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agents.prompts import (
    PIPELINE_SYSTEM_PROMPT,
    SKILL_SYSTEM_PROMPT,
    build_fix_prompt,
    build_initial_skill_prompt,
    build_pipeline_prompt,
)
from agents.text_utils import extract_code, strip_optional_fence
from config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    """Provider-agnostic: settings.llm_provider selects anthropic/openai/ollama/...
    (the matching langchain-<provider> integration package must be installed)."""
    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.llm_provider,
        api_key=settings.api_key or None,
        temperature=settings.temperature,
        max_tokens=16000,
    )


def write_skill(llm: BaseChatModel, retrieved_papers: str) -> str:
    prompt = build_initial_skill_prompt(retrieved_papers)
    response = llm.invoke(
        [("system", SKILL_SYSTEM_PROMPT), ("human", prompt)]
    )
    # .text handles both plain-string content and content-block lists (e.g. a
    # reasoning model's text+thinking blocks), regardless of provider.
    return strip_optional_fence(response.text)


def generate_pipeline(llm: BaseChatModel, skill_md: str) -> str:
    prompt = build_pipeline_prompt(skill_md)
    response = llm.invoke(
        [("system", PIPELINE_SYSTEM_PROMPT), ("human", prompt)]
    )
    return extract_code(response.text)


def fix_pipeline(llm: BaseChatModel, previous_code: str, error_message: str) -> str:
    prompt = build_fix_prompt(previous_code, error_message)
    response = llm.invoke(
        [("system", PIPELINE_SYSTEM_PROMPT), ("human", prompt)]
    )
    return extract_code(response.text)
