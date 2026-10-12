"""Tools the swarm agents can call (via langchain's create_agent, agents/swarm_agent.py).

    search_corpus             RAG over the literature corpus (corpus/retrieve.py), honouring
                              settings.retrieval_filter
    web_search                Tavily web search (needs TAVILY_API_KEY; omitted without it)
    request_hypothesis_check  conversation turns only: asks the HYPOTHESIS CHECKER to run an
                              analysis. The tool itself only records the request; the talk node
                              (orchestration/graph.py) hands off to the checker node after the turn.

Tools are built per agent invocation around a ToolRecorder, which keeps what the agent did
(queries, papers seen, check requested) for the ledger. Whatever a tool returns goes only into
the calling agent's own context -- never onto the blackboard.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool

from config import Settings
from corpus.retrieve import format_for_prompt, retrieve


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[tools] {msg}", file=sys.stderr)


@dataclass
class ToolRecorder:
    rag_queries: list[str] = field(default_factory=list)
    rag_papers: list[str] = field(default_factory=list)
    web_queries: list[str] = field(default_factory=list)
    check_question: str | None = None
    checks_left: int = 0

    def stats(self) -> dict:
        return {
            "rag_queries": list(self.rag_queries),
            "rag_papers": list(dict.fromkeys(self.rag_papers)),
            "web_queries": list(self.web_queries),
        }


def make_search_corpus_tool(settings: Settings, recorder: ToolRecorder) -> BaseTool:
    @tool
    def search_corpus(query: str, k: int = settings.rag_tool_k) -> str:
        """Search the curated corpus of landslide-mapping publications. Returns, for each of the
        k most relevant papers, its bracketed paper id, a summary of its methods and datasets,
        and an excerpt of its full text. Cite papers by their bracketed id."""
        k = max(1, min(int(k), 2 * settings.rag_tool_k))
        recorder.rag_queries.append(query)
        try:
            docs = retrieve(query, settings, k=k)
        except Exception as exc:  # noqa: BLE001 - a failed search must not sink the turn
            return f"corpus search failed: {exc!r}"
        if not docs:
            return "No paper in the corpus matched this query."
        recorder.rag_papers.extend(d.metadata.get("paper_id", "unknown") for d in docs)
        return format_for_prompt(docs, settings, char_cap=settings.rag_tool_char_cap)

    return search_corpus


def make_web_search_tool(settings: Settings, recorder: ToolRecorder) -> BaseTool | None:
    """Tavily web search, or None (with a note on stderr) if it can't be built -- off in
    config, missing langchain-tavily or TAVILY_API_KEY -- so agents just go without it."""
    if not settings.web_search:
        return None
    if not os.environ.get("TAVILY_API_KEY"):
        _warn_once("web_search unavailable: TAVILY_API_KEY is not set in .env")
        return None
    try:
        from langchain_tavily import TavilySearch

        tavily = TavilySearch(max_results=settings.web_search_max_results)
    except Exception as exc:  # noqa: BLE001 - any setup failure just disables the tool
        _warn_once(f"web_search unavailable ({type(exc).__name__}: {exc})")
        return None

    @tool
    def web_search(query: str) -> str:
        """Search the web (e.g. for the exact formulation of a technique, or a library's API).
        Returns the top results with their URLs and content snippets."""
        recorder.web_queries.append(query)
        try:
            return str(tavily.invoke({"query": query}))
        except Exception as exc:  # noqa: BLE001 - a failed search must not sink the turn
            return f"web search failed: {exc!r}"

    return web_search


def make_check_request_tool(recorder: ToolRecorder) -> BaseTool:
    @tool
    def request_hypothesis_check(question: str) -> str:
        """Ask the HYPOTHESIS CHECKER to test a hypothesis on the agents' validation predictions
        (every agent's predicted masks and the ground-truth masks of this round). Ask a precise,
        answerable question. The checker runs the analysis after your turn and posts its numbers
        on the blackboard for everyone. At most one request per turn."""
        if recorder.check_question is not None:
            return "You already requested a check this turn; only one request per turn is allowed."
        if recorder.checks_left <= 0:
            return "No hypothesis checks are left this round; continue without one."
        recorder.check_question = question
        return (
            "Request accepted. The HYPOTHESIS CHECKER will post its results on the blackboard "
            "right after your turn. Now write your post (or PASS)."
        )

    return request_hypothesis_check


def agent_tools(settings: Settings, recorder: ToolRecorder, conversation: bool) -> list[BaseTool]:
    """The tools of a swarm agent: corpus + web search always (when available), plus the check
    request during conversation turns."""
    tools: list[BaseTool] = []
    if settings.use_rag:
        tools.append(make_search_corpus_tool(settings, recorder))
    web = make_web_search_tool(settings, recorder)
    if web is not None:
        tools.append(web)
    if conversation:
        tools.append(make_check_request_tool(recorder))
    return tools
