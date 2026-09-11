"""Peer review / PSO-style skill update, adapted from AgentPSO (Hwang et al., 2026).

Topology: global-best -- every agent's neighbourhood is the whole rest of the swarm,
and every agent is steered toward one swarm-wide global-best (per
documentation/project_implementation_steps.md's recommendation to start there;
local/neighbourhood topologies are future work).

Three semantic operators replace AgentPSO's numeric velocity update:

    d_i^t   = Reflect(skill_i, peer_observation_i)
    v_i^t+1 = VelocityUpdate(v_i^t, d_i^t, skill_i, p_best_i, g_best)
    s_i^t+1 = SkillUpdate(skill_i, v_i^t+1)

Peer observation deliberately excludes peer skill.md text -- only each peer's Dice/IoU
and actual pipeline.py are shown. This mirrors AgentPSO's own finding (Section
4.1, Table 6) that reflecting on a peer's *behavior/trace* rather than copying their
*instruction text* produces better, more transferable updates. Personal-best and
global-best skill text, by contrast, ARE passed directly into VelocityUpdate -- those
are the swarm's already-vetted positions (not a private same-round peer attempt), so
drawing on them directly is exactly the point of PSO.

Reflect is also grounded in the literature corpus every round (not just round 0): before
reflecting, we retrieve a handful of papers using the agent's current approach + this
round's peer comparison as the query, and hand that literature to the Reflect call so it
can explain a performance gap by citing a specific published technique instead of just
guessing -- matching the proposal's "ground each pipeline revision in a specific
published technique." Retrieval happens once per agent per round, before Reflect, rather
than a two-pass "reflect, then search, then re-reflect" loop -- cheaper (no extra LLM
call to formulate a search query) and the query is arguably more concrete anyway, since
it's anchored on the actual code/score gap rather than an already-abstracted reflection.

With no peers (n_agents == 1) this is a no-op: reflect_and_update returns the agent's
skill unchanged, and orchestration/graph.py carries the previous round forward instead
of wastefully re-running an identical pipeline.
"""
from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from agents.text_utils import strip_optional_fence
from config import Settings
from corpus.retrieve import format_for_prompt, retrieve
from orchestration.state import AgentState

_MAX_PEER_CODE_CHARS = 6000
_MAX_QUERY_SKILL_CHARS = 1500
_MAX_QUERY_PEER_CODE_CHARS = 800

# Paper ids are Scopus EIDs, e.g. "2-s2.0-105000159569" -- the same id scheme used as
# both the corpus JSON filenames and the "[paper_id]" citation style skill.md/prompts
# already use throughout (agents/prompts.py, corpus/retrieve.py::format_for_prompt).
_CITATION_RE = re.compile(r"2-s2\.0-\d+")


def extract_cited_papers(text: str) -> list[str]:
    """Paper ids actually cited in a piece of generated text (e.g. a skill.md), in
    first-seen order, deduplicated. This is the real citation trail for a report --
    distinct from which papers were merely retrieved/shown to the agent that round."""
    seen: list[str] = []
    for match in _CITATION_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen

REFLECT_SYSTEM_PROMPT = """\
You are a research agent developing a landslide detection/mapping pipeline. You just \
finished a round: your pipeline ran and scored some Dice/IoU. You can also see what your \
peers' pipelines actually did this round (their code and score), but NOT their skill.md \
strategy notes -- you must infer the reasoning from behavior, not copy their words.

Compare your approach to each peer's: identify concrete, specific mechanisms (a \
particular preprocessing step, engineered feature, model choice, or post-processing \
step) that plausibly explain any performance gap, in either direction. Do not just say \
"peer X scored higher, adopt peer X" -- explain WHY a specific technique likely helped \
or hurt, so the lesson generalizes. If your own approach already handles something \
better than every peer, say so explicitly -- this is not one-directional copying.

You are also given a handful of papers retrieved from the literature corpus for this \
round, relevant to your current approach and this comparison. Where one of them actually \
supports or explains a specific mechanism you're reasoning about, cite it inline using \
its bracketed id exactly as it appears (e.g. [2-s2.0-12345678901]), the same citation \
style skill.md already uses. Do not force a citation where none of the retrieved papers \
are actually relevant -- an ungrounded reflection is better than a fabricated citation.

Output a short, concrete self-reflective direction (a few bullet points), not a full \
plan and not code."""

VELOCITY_SYSTEM_PROMPT = """\
You maintain one landslide-mapping research agent's "semantic velocity": a running, \
concise set of directives for how its skill.md should keep evolving. You are given the \
previous velocity, a fresh self-reflective direction from this round, the agent's \
current skill, its own personal-best skill, and the swarm's global-best skill so far.

Synthesize ONE updated velocity: a concise, concrete list of revision directives (what \
to keep from the current skill, what to adopt from the personal-best/global-best skill, \
what to change per the fresh reflection, what to discard because it underperformed). \
Resolve conflicts explicitly (e.g. if the reflection contradicts the previous velocity, \
say which wins and why). Output ONLY the directives, not the skill.md itself, not code."""

SKILL_UPDATE_SYSTEM_PROMPT = """\
You rewrite a research agent's skill.md -- its durable, accumulated strategy for a \
landslide-mapping pipeline -- by applying a given set of revision directives (a semantic \
velocity). Build on the current skill.md rather than starting from scratch: keep what \
the directives don't ask you to change, including existing literature citations, and \
apply the directives concretely (state the specific new preprocessing/feature/model/ \
post-processing choices, and why). Output the full revised skill.md, in the same style \
as the input (concrete and actionable, citing paper ids where relevant)."""


def _format_score(value: float | None) -> str:
    return "N/A (run failed)" if value is None else f"{value:.4f}"


def _format_peer_observations(neighbourhood: list[AgentState]) -> str:
    blocks = []
    for peer in neighbourhood:
        code = peer.last_code or "(no code recorded -- last run failed)"
        if len(code) > _MAX_PEER_CODE_CHARS:
            code = code[:_MAX_PEER_CODE_CHARS] + "\n# ...[truncated]"
        blocks.append(
            f"Peer agent_{peer.agent_idx}: "
            f"Dice={_format_score(peer.last_dice)}, IoU={_format_score(peer.last_iou)}\n"
            f"```python\n{code}\n```"
        )
    return "\n\n".join(blocks)


def _build_grounding_query(agent: AgentState, neighbourhood: list[AgentState]) -> str:
    """Retrieval query for this round's grounding: the agent's own current approach
    plus a digest of what peers actually did, so the corpus search is anchored on the
    concrete technique gap this round's Reflect call needs to explain -- no extra LLM
    call needed to formulate the query."""
    own = (agent.skill_md or "(no strategy yet)")[:_MAX_QUERY_SKILL_CHARS]
    peers = "\n".join(
        f"peer_{p.agent_idx} (Dice={_format_score(p.last_dice)}): "
        f"{(p.last_code or '(no code)')[:_MAX_QUERY_PEER_CODE_CHARS]}"
        for p in neighbourhood
    )
    return f"Current approach:\n{own}\n\nPeer approaches this round:\n{peers}"


def reflect(
    llm: BaseChatModel,
    skill_md: str,
    last_dice: float | None,
    last_iou: float | None,
    neighbourhood: list[AgentState],
    grounding: str,
) -> str:
    prompt = f"""\
Your current skill.md:
{skill_md}

Your last result: Dice={_format_score(last_dice)}, IoU={_format_score(last_iou)}

Peer observations this round (their code + score):
{_format_peer_observations(neighbourhood)}

Retrieved literature relevant to this round's comparison (paper id, methods, datasets, novelty):
{grounding}"""
    response = llm.invoke([("system", REFLECT_SYSTEM_PROMPT), ("human", prompt)])
    return response.text.strip()


def velocity_update(
    llm: BaseChatModel,
    prev_velocity: str,
    reflection: str,
    skill_md: str,
    p_best_skill: str,
    g_best_skill: str | None,
) -> str:
    prompt = f"""\
Previous velocity (revision directives from last round):
{prev_velocity or "(none yet -- this is the first update)"}

Fresh self-reflective direction from this round:
{reflection}

Your current skill.md:
{skill_md}

Your personal-best skill.md so far:
{p_best_skill or "(same as current -- no better round yet)"}

Swarm global-best skill.md so far:
{g_best_skill or "(no global-best recorded yet)"}"""
    response = llm.invoke([("system", VELOCITY_SYSTEM_PROMPT), ("human", prompt)])
    return response.text.strip()


def skill_update(llm: BaseChatModel, skill_md: str, velocity: str) -> str:
    prompt = f"""\
Your current skill.md:
{skill_md}

Revision directives (velocity) to apply:
{velocity}"""
    response = llm.invoke([("system", SKILL_UPDATE_SYSTEM_PROMPT), ("human", prompt)])
    return strip_optional_fence(response.text)


def reflect_and_update(
    llm: BaseChatModel,
    agent: AgentState,
    neighbourhood: list[AgentState],
    g_best_skill: str | None,
    settings: Settings,
) -> tuple[str, str, bool, list[str]]:
    """Run Reflect -> VelocityUpdate -> SkillUpdate. Returns (new_skill_md, new_velocity,
    changed, retrieved_paper_ids). changed=False (and retrieved_paper_ids=[]) only when
    there are no peers to reflect against."""
    if not neighbourhood:
        return agent.skill_md, agent.velocity, False, []

    query = _build_grounding_query(agent, neighbourhood)
    docs = retrieve(query, settings)
    grounding = format_for_prompt(docs, settings)
    retrieved_paper_ids = [d.metadata.get("paper_id", "unknown") for d in docs]

    d = reflect(llm, agent.skill_md, agent.last_dice, agent.last_iou, neighbourhood, grounding)
    v = velocity_update(
        llm, agent.velocity, d, agent.skill_md, agent.p_best_skill, g_best_skill
    )
    s = skill_update(llm, agent.skill_md, v)
    return s, v, True, retrieved_paper_ids
