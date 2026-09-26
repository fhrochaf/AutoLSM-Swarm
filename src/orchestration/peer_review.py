"""Peer review / PSO-style skill update, adapted from AgentPSO (Hwang et al., 2026).

Topology: global-best -- every agent's neighbourhood is the whole rest of the swarm,
and every agent is steered toward one swarm-wide global-best (per
documentation/project_implementation_steps.md's recommendation to start there;
local/neighbourhood topologies are future work).

Four semantic operators replace numeric velocity update:

    d_i^t          = Reflect(skill_i, own_observation_i, peer_observation_i)
    d_i^t (grounded) = GroundReflection(d_i^t, retrieved_lit_i)
    v_i^t+1        = VelocityUpdate(v_i^t, d_i^t (grounded), skill_i, p_best_i, g_best)
    s_i^t+1        = SkillUpdate(skill_i, v_i^t+1)

Peer observation deliberately excludes peer skill.md text -- only each peer's Dice/IoU
and actual pipeline.py are shown.

Reflect runs first, purely from behavior (own code+score alongside each peer's code+score,
shown as parallel blocks so the comparison is code-to-code, not skill-text-to-code) and closes
with one explicit "GROUNDING QUERY: ..." line naming the mechanism it's least
sure about. That line is used as the retrieval query. GroundReflection is a second LLM pass
that reads the draft reflection back together with the retrieved
papers and revises it: confirming, sharpening, or contradicting its own hypotheses and
attaching citations where a paper actually bears on one.

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

settings = Settings()

_MAX_PEER_CODE_CHARS = settings.full_text_char_cap

# Paper ids are Scopus EIDs, e.g. "2-s2.0-105000159569" -- the same id scheme used as
# both the corpus JSON filenames and the "[paper_id]" citation style skill.md/prompts
# already use throughout (agents/prompts.py, corpus/retrieve.py::format_for_prompt).
_CITATION_RE = re.compile(r"2-s2\.0-\d+")

# Matches the trailing "GROUNDING QUERY: ..." line Reflect is asked to emit.
_GROUNDING_QUERY_RE = re.compile(r"(?im)^GROUNDING QUERY:\s*(.+?)\s*$")


def extract_cited_papers(text: str) -> list[str]:
    """Paper ids actually cited in a piece of generated text (e.g. a skill.md), in
    first-seen order, deduplicated. This is the real citation trail for a report --
    distinct from which papers were merely retrieved/shown to the agent that round."""
    seen: list[str] = []
    for match in _CITATION_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen

REFLECT_SYSTEM_PROMPT =f"""\
You are a research agent developing a landslide detection/mapping pipeline. Below: your \
skill.md, then your pipeline.py + score this round, then each peer's pipeline.py + score.

Instruction:
- Compare your code to each peer's, mechanism by mechanism -- not "peer X scored higher, \
adopt peer X", but WHY a specific technique likely helped or hurt, so the lesson generalizes.
- If your code already does something better than every peer's, say so.
- Return only a few bullet points describing to which direction your skill.md should be updated
to achieve beter score."""

REFLECT_SYSTEM_PROMPT_2 =f"""\Then end with exactly one line:
GROUNDING QUERY: <a concrete question a literature search should resolve>"""

ENRICHED_REFLECTION_SYSTEM_PROMPT = """\
You are a research agent developing a landslide detection/mapping pipeline, revisiting a self-reflective comparison you just wrote \
against your peers. You flagged mechanisms you were unsure about, and are \
now given a handful of papers retrieved from the literature corpus targeting exactly that \
question.

Revise your reflection in light of this literature: where a retrieved paper actually \
confirms, sharpens, or contradicts one of your hypotheses, say so explicitly and cite it \
inline using its bracketed id exactly as it appears (e.g. [2-s2.0-12345678901]), the same \
citation style skill.md already uses. Do not force a citation onto a point none of the \
retrieved papers actually bear on -- an ungrounded (but still concrete) hypothesis is \
better than a fabricated or tenuous citation. If none of the retrieved papers resolve your \
open question, say so plainly and leave that point reasoned from behavior alone.

Output the final, revised self-reflective direction (a few bullet points) -- not a full \
plan, not code, and no GROUNDING QUERY line this time; that step is done."""

VELOCITY_SYSTEM_PROMPT = """\
You maintain one landslide-mapping research agent's "semantic velocity": directives for \
how its skill.md -- the strategy of how to elaborate a pipeline.py script desgined to map landslides -- should evolve. \
Below: the previous velocity, this round's self-reflective direction, your current skill, \
your personal-best skill, the global-best skill.

Instruction:
- Combine the previous velocity, the fresh direction, and lessons from the personal-best \
and global-best skills.
- Focus on generalizable improvements, not one-off fixes.
- Do not copy the personal-best or global-best skill directly.
- Do not just converge it into a copy of another agent.
- Return a concise natural-language velocity."""

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


def _format_observations(
    own_code: str,
    own_dice: float | None,
    own_iou: float | None,
    neighbourhood: list[AgentState],
) -> str:
    entries = [("[Your pipeline]", own_code, own_dice, own_iou)] + [
        (f"[Peer agent_{peer.agent_idx}]", peer.last_code, peer.last_dice, peer.last_iou)
        for peer in neighbourhood
    ]
    blocks = []
    for label, code, dice, iou in entries:
        code = code or "(no code recorded -- last run failed)"
        if len(code) > _MAX_PEER_CODE_CHARS:
            code = code[:_MAX_PEER_CODE_CHARS] + "\n# ...[truncated]"
        blocks.append(f"{label}: Dice={_format_score(dice)}, IoU={_format_score(iou)}\n```python\n{code}\n```")
    return "\n\n".join(blocks)


def extract_grounding_query(reflection: str) -> tuple[str, str]:
    """Split a Reflect response into (comparison_body, grounding_query). The query is
    the trailing "GROUNDING QUERY: ..." line Reflect is asked to emit -- parsed out
    mechanically rather than via a second LLM call, since formulating it is a small
    enough job that Reflect can do as part of the same response. Falls back to using
    the full reflection as both if the model didn't follow the format."""
    match = _GROUNDING_QUERY_RE.search(reflection)
    if not match:
        return reflection, reflection
    body = reflection[: match.start()].strip()
    return body or reflection, match.group(1).strip()


def reflect(
    llm: BaseChatModel,
    skill_md: str,
    last_code: str,
    last_dice: float | None,
    last_iou: float | None,
    neighbourhood: list[AgentState],
    enriched_reflection: bool = False
) -> str:
    prompt = f"""\
Your current skill.md:
{skill_md}

This round's outputs -- your pipeline.py and score, then each peer's, for comparison:
{_format_observations(last_code, last_dice, last_iou, neighbourhood)}"""

    reflect_system_promt = f"{REFLECT_SYSTEM_PROMPT} {REFLECT_SYSTEM_PROMPT_2 if enriched_reflection else ""}"

    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", reflect_system_promt), ("human", prompt)]
    )
    return response.text.strip()


def enrich_reflection(llm: BaseChatModel, reflection: str, grounding: str) -> str:
    prompt = f"""\
Your first-pass reflection this round:
{reflection}

Retrieved literature addressing your flagged open question (paper id, methods, datasets, novelty):
{grounding}"""
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", ENRICHED_REFLECTION_SYSTEM_PROMPT), ("human", prompt)]
    )
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
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", VELOCITY_SYSTEM_PROMPT), ("human", prompt)]
    )
    return response.text.strip()


def skill_update(llm: BaseChatModel, skill_md: str, velocity: str) -> str:
    prompt = f"""\
Your current skill.md:
{skill_md}

Revision directives (velocity) to apply:
{velocity}"""
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", SKILL_UPDATE_SYSTEM_PROMPT), ("human", prompt)]
    )
    return strip_optional_fence(response.text)


def reflect_and_update(
    llm: BaseChatModel,
    agent: AgentState,
    neighbourhood: list[AgentState],
    g_best_skill: str | None,
    settings: Settings,
    enriched_reflection: bool = False,
) -> tuple[str, str, bool, list[str]]:
    """Run Reflect -> GroundReflection -> VelocityUpdate -> SkillUpdate. Returns
    (new_skill_md, new_velocity, changed, retrieved_paper_ids). changed=False (and
    retrieved_paper_ids=[]) only when there are no peers to reflect against."""
    if not neighbourhood:
        return agent.skill_md, agent.velocity, False, []

    reflection = reflect(llm, agent.skill_md, agent.last_code, agent.last_dice, agent.last_iou, neighbourhood, enriched_reflection)
    if enriched_reflection:
        draft, query = extract_grounding_query(reflection)
        docs = retrieve(query, settings)
        grounding = format_for_prompt(docs, settings)
        reflection = enrich_reflection(llm, draft, grounding)

    retrieved_paper_ids = [d.metadata.get("paper_id", "unknown") for d in (docs or [])]
    v = velocity_update(llm, agent.velocity, reflection, agent.skill_md, agent.p_best_skill, g_best_skill)
    s = skill_update(llm, agent.skill_md, v)
    return s, v, True, retrieved_paper_ids
