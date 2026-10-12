"""The swarm agents, built with langchain's create_agent().

Every LLM actor in the swarm -- the N swarm agents, the OBSERVER and the HYPOTHESIS CHECKER --
is a create_agent() graph with its own tools, wrapped in the same middleware: provider retries
(ModelRetryMiddleware) and a cap on tool calls per invocation (ToolCallLimitMiddleware).

A swarm agent is invoked for three things, always with the same system prompt
(prompts.yaml `agent.system`) and a context rebuilt fresh from the swarm state
(agents/context.py):

    build_pipeline   write the next pipeline.py (+ the private notebook it carries forward)
    take_turn        one turn on the blackboard: a post, PASS, and maybe a check request
    fix_pipeline     the debug loop (agents/runner.py) -- no tools, just the fix
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from agents.prompts import agent_system_prompt, render
from agents.text_utils import cap_text, extract_code, extract_notebook
from agents.tools import ToolRecorder, agent_tools
from config import Settings

_LLMS: dict[tuple, BaseChatModel] = {}

# `request_hypothesis_check(question="...")` written into a post as text instead of called.
_WRITTEN_CHECK_RE = re.compile(
    r"`?request_hypothesis_check\(\s*(?:question\s*=\s*)?([\"'])(.+?)\1\s*\)`?", re.DOTALL
)


def _chat_model(
    provider: str, model: str, api_key: str, max_tokens: int, model_kwargs: dict | None = None
) -> BaseChatModel:
    params: dict = {"max_tokens": max_tokens}
    if provider == "openai":
        # Chat Completions rejects function tools together with reasoning on recent OpenAI models;
        # the Responses API supports both.
        params["use_responses_api"] = True
    params.update(model_kwargs or {})
    params = {k: v for k, v in params.items() if v is not None}  # None removes a default
    key = (provider, model, json.dumps(params, sort_keys=True, default=str))
    if key not in _LLMS:
        _LLMS[key] = init_chat_model(model=model, model_provider=provider, api_key=api_key or None, **params)
    return _LLMS[key]


def make_llm(settings: Settings, role: int) -> BaseChatModel:
    """The chat model of a non-agent role (config.py): 2 = observer, 3 = checker."""
    return _chat_model(
        getattr(settings, f"llm_provider_{role}"), getattr(settings, f"model_name_{role}"),
        getattr(settings, f"api_key_{role}"), getattr(settings, f"max_tokens_{role}"),
    )


def make_agent_llm(settings: Settings, agent_idx: int) -> BaseChatModel:
    """The chat model swarm agent `agent_idx` runs on (its group in settings.agents)."""
    g = settings.agent_models()[agent_idx]
    return _chat_model(g.llm_provider, g.model_name, g.api_key, g.max_tokens, g.model_kwargs)


def invoke_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    messages: list[BaseMessage],
    settings: Settings,
    name: str,
    max_tool_calls: int | None = None,
) -> list[BaseMessage]:
    """Run one create_agent() invocation to its final answer and return the whole message list.

    If the agent keeps calling tools past its budget (the tool-call limit middleware answers
    every extra call with an error, but a stubborn model can still loop), the recursion limit
    stops it and the model is asked once more, without tools, for its final answer."""
    max_tool_calls = max_tool_calls or settings.agent_max_tool_calls
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        middleware=[
            ModelRetryMiddleware(max_retries=settings.llm_retry_attempts, on_failure="error"),
            ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
        ],
        name=name,
    )
    try:
        result = agent.invoke(
            {"messages": messages}, config={"recursion_limit": 4 * max_tool_calls + 10}
        )
        return result["messages"]
    except GraphRecursionError:
        final = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
            [SystemMessage(system_prompt), *messages,
             HumanMessage("You are out of tool calls. Give your final answer now, without tools.")]
        )
        return [*messages, final]


def final_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.text.strip():
            return m.text.strip()
    return ""


def tool_calls_made(messages: list[BaseMessage]) -> int:
    return sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))


def _section(kind: str, body: str) -> str:
    # One message per section, behind a delimiter line that markdown content cannot produce
    # (the run viewer in frontend/ splits transcripts on it).
    return f"===== [{kind}] =====\n{body.strip()}"


def transcript(messages: list[BaseMessage], system_prompt: str | None = None, result_cap: int = 4000) -> str:
    """A readable log of one invocation: the system prompt, every input message, every tool call
    with its arguments and a capped view of its result, and the final answer."""
    parts = [_section("system", system_prompt)] if system_prompt else []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(_section("input", m.text))
        elif isinstance(m, AIMessage):
            for call in m.tool_calls:
                parts.append(_section(f"tool call: {call['name']}", json.dumps(call["args"], indent=2, ensure_ascii=False)))
            if m.text.strip():
                parts.append(_section("answer", m.text))
        elif isinstance(m, ToolMessage):
            parts.append(_section(f"tool result: {m.name}", cap_text(str(m.content), result_cap)))
    return "\n\n".join(parts) + "\n"


@dataclass
class BuildResult:
    code: str
    notebook: str
    transcript: str
    stats: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    post: str | None  # None = PASS
    check_question: str | None
    transcript: str
    stats: dict = field(default_factory=dict)


def build_pipeline(agent_idx: int, n_agents: int, messages: list[BaseMessage], settings: Settings) -> BuildResult:
    recorder = ToolRecorder()
    system = agent_system_prompt(agent_idx, n_agents)
    out = invoke_agent(
        make_agent_llm(settings, agent_idx), agent_tools(settings, recorder, conversation=False),
        system, messages, settings, name=f"agent_{agent_idx}",
    )
    text = final_text(out)
    return BuildResult(
        code=extract_code(text),
        notebook=extract_notebook(text, settings.notebook_char_cap),
        transcript=transcript(out, system),
        stats={"tool_calls": tool_calls_made(out), **recorder.stats()},
    )


def take_turn(
    agent_idx: int, n_agents: int, messages: list[BaseMessage], settings: Settings, checks_left: int
) -> TurnResult:
    recorder = ToolRecorder(checks_left=checks_left)
    system = agent_system_prompt(agent_idx, n_agents)
    out = invoke_agent(
        make_agent_llm(settings, agent_idx), agent_tools(settings, recorder, conversation=True),
        system, messages, settings, name=f"agent_{agent_idx}",
    )
    text = final_text(out)
    check_question = recorder.check_question
    # Some models write the call into their post instead of calling the tool; honour it once.
    written = _WRITTEN_CHECK_RE.search(text)
    if written:
        text = (text[: written.start()] + text[written.end():]).strip()
        if check_question is None and checks_left > 0:
            check_question = written.group(2).strip()
    passed = not text or text.strip().strip(".").upper() == "PASS"
    return TurnResult(
        post=None if passed else text,
        check_question=check_question,
        transcript=transcript(out, system),
        stats={"tool_calls": tool_calls_made(out), **recorder.stats()},
    )


def fix_pipeline(
    agent_idx: int, n_agents: int, settings: Settings, previous_code: str, error_message: str
) -> tuple[str, str]:
    """One debug-loop fix by the agent that wrote the pipeline (no tools: the fix should be the
    smallest change that makes the error go away). Returns (fixed code, transcript)."""
    system = agent_system_prompt(agent_idx, n_agents)
    messages = [HumanMessage(render("agent.fix_user", previous_code=previous_code, error_message=error_message))]
    llm = make_agent_llm(settings, agent_idx).with_retry(stop_after_attempt=settings.llm_retry_attempts)
    response = llm.invoke([SystemMessage(system), *messages])
    return extract_code(response.text), transcript([*messages, response], system)
