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

from agents.prompts import DATASET_DESCRIPTION
from agents.text_utils import strip_optional_fence
from config import Settings
from corpus.retrieve import format_for_prompt, retrieve
from orchestration.state import AgentState, VelocityRecord

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
skill.md (which describes the strategy to implement code for the mapping pipeline),
then your pipeline.py (generated after that strategy) + score this round,
then each peer's pipeline.py + score.

Instruction:
- Compare your code to each peer's, mechanism by mechanism -- not "peer X scored higher, \
adopt peer X", but WHY a specific technique likely helped or hurt, so the lesson generalizes. \
A mechanism can be architectural (a technique, layer, or preprocessing step) or a specific \
declared hyperparameter/threshold value (e.g. "my dropout=0.5 vs. peer's 0.2") -- compare both.
- If your code already does something better than every peer's, say so.
- Return only a few bullet points describing candidate directions your skill.md could be \
updated in to achieve a better score. This is diagnosis, not the decision -- the velocity \
update step chooses which single one to actually act on."""

REFLECT_SYSTEM_PROMPT_2 = """\
Then end with exactly one line:
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
Below: the previous velocity, the recent trajectory (past velocities and the score change \
each one produced), this round's self-reflective direction, your current skill, \
your personal-best skill, the global-best skill.

Instruction:
- Use the recent trajectory as momentum: keep pushing in a direction that raised the score, \
and back off or change course where it lowered it or the run failed. Small score changes \
(under ~0.01 Dice) may be noise, not evidence -- do not over-credit or over-blame a \
directive for them.
- Choose EXACTLY ONE change to make this round -- either ONE structural mechanism (one \
architecture/preprocessing/loss-family/post-processing technique to add, replace, or \
remove) OR ONE parametric change (one declared hyperparameter moved to a new value). \
Never both, never several of either: a velocity that bundles multiple simultaneous \
changes makes it impossible to tell which one actually caused next round's score to move, \
which breaks the momentum rule above.
- If the chosen change is parametric: name the exact parameter as it is declared in the \
current skill.md's `## Hyperparameters` section, and state it as exactly \
`CHANGE <param_name>: <current value> -> <new value>` on its own line, using the value \
actually shown in skill.md as <current value> -- never a guessed or rounded one.
- If the reflection, trajectory, personal-best, or global-best skill suggest more than \
one promising direction, name the others briefly as deferred candidates for a future \
round, clearly separated from the one directive being acted on now.
- Begin the velocity with one line "FROM: <style of the current skill> -> TO: <where the \
skill is being pushed>", then the one chosen directive, then any deferred candidates.
- Combine the previous velocity, the fresh direction, and lessons from the personal-best \
and global-best skills when choosing which single change to make.
- Focus on generalizable improvements, not one-off fixes.
- Do not copy the personal-best or global-best skill directly.
- Do not just converge it into a copy of another agent."""

SKILL_UPDATE_SYSTEM_PROMPT = """\
You rewrite a research agent's skill.md -- its durable, accumulated strategy for a \
landslide-mapping pipeline -- by applying a given revision directive (a semantic \
velocity). The velocity names EXACTLY ONE change to make this round -- either a \
structural mechanism, or a parametric change in the form \
`CHANGE <param_name>: <old> -> <new>` -- and may also name other candidate directions it \
explicitly deferred to a future round. Apply only the one change being acted on now; \
ignore the deferred candidates entirely, they are not part of this update.

Build on the current skill.md rather than starting from scratch: keep everything the \
directive doesn't ask you to change, including existing literature citations and every \
other declared value in the `## Hyperparameters` section. Apply the one directive \
concretely:
- Structural change: state the specific new preprocessing/feature/model/post-processing \
choice, and why.
- Parametric change: update ONLY that parameter's line in `## Hyperparameters` to the new \
value the directive gives (never a different number), and adjust any prose elsewhere \
that names the old value so the document stays internally consistent.

Every skill.md you output MUST still end with a `## Hyperparameters` section listing \
every tunable numeric/categorical parameter the pipeline uses, one per line as \
`- <param_name> = value  # short reason` -- carry this section forward, adding an \
entry if the current skill.md doesn't yet declare a parameter the pipeline already \
depends on. Output the full revised skill.md, in the same style as the input (concrete \
and actionable, citing paper ids where relevant)."""


def _format_score(value: float | None) -> str:
    if value is None:
        return "N/A (run failed)"
    if value in (float("inf"), float("-inf")):
        return "N/A (no best yet)"
    return f"{value:.4f}"


def _best_score(dices: list[float | None]) -> float | None:
    known = [d for d in dices if d is not None]
    return max(known) if known else None


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
{DATASET_DESCRIPTION}

Your current skill.md:
{skill_md}

This round's outputs -- your pipeline.py and score, then each peer's, for comparison:
{_format_observations(last_code, last_dice, last_iou, neighbourhood)}"""

    reflect_system_promt = REFLECT_SYSTEM_PROMPT
    if enriched_reflection:
        reflect_system_promt = f"{REFLECT_SYSTEM_PROMPT}\n{REFLECT_SYSTEM_PROMPT_2}"

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


def _format_trajectory(history: list[VelocityRecord]) -> str:
    if not history:
        return "(no earlier velocities yet)"
    lines = []
    for rec in history:
        before = _format_score(rec.dice_before)
        if not rec.outcome_recorded:
            outcome = "outcome not yet known"
        elif rec.dice_after is None:
            outcome = "the resulting run FAILED"
        elif rec.dice_before is None:
            outcome = f"resulting Dice={rec.dice_after:.4f}"
        else:
            outcome = f"resulting Dice={rec.dice_after:.4f} ({rec.dice_after - rec.dice_before:+.4f})"
        lines.append(f"- Round {rec.round}: applied to a skill scoring Dice={before} -> {outcome}\n  {rec.velocity}")
    return "\n".join(lines)


def velocity_update(
    llm: BaseChatModel,
    prev_velocity: str,
    reflection: str,
    skill_md: str,
    p_best_skill: str,
    g_best_skill: str | None,
    history: list[VelocityRecord] | None = None,
    current_dice: float | None = None,
    p_best_score: float | None = None,
    g_best_score: float | None = None,
    best_this_round: float | None = None,
) -> str:
    prompt = f"""\
Previous velocity (revision directives from last round):
{prev_velocity or "(none yet -- this is the first update)"}

Recent trajectory (oldest first) -- what earlier velocities did to the score:
{_format_trajectory(history or [])}

Your current skill.md scores Dice={_format_score(current_dice)}.
Best score this round, across yourself and every peer: Dice={_format_score(best_this_round)}.

Fresh self-reflective direction from this round:
{reflection}

Your current skill.md:
{skill_md}

Your personal-best skill.md so far (Dice={_format_score(p_best_score)}):
{p_best_skill or "(same as current -- no better round yet)"}

Swarm global-best skill.md so far (Dice={_format_score(g_best_score)}):
{g_best_skill or "(no global-best recorded yet)"}"""
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", VELOCITY_SYSTEM_PROMPT), ("human", prompt)]
    )
    return response.text.strip()


def skill_update(llm: BaseChatModel, skill_md: str, velocity: str) -> str:
    prompt = f"""\
{DATASET_DESCRIPTION}

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
    g_best_score: float,
    settings: Settings,
    enriched_reflection: bool = False,
) -> tuple[str, str, bool, list[str], list[VelocityRecord]]:
    """Run Reflect -> GroundReflection -> VelocityUpdate -> SkillUpdate. Returns
    (new_skill_md, new_velocity, changed, retrieved_paper_ids, new_velocity_history).
    changed=False (and retrieved_paper_ids=[]) only when there are no peers to reflect
    against."""
    if not neighbourhood:
        return agent.skill_md, agent.velocity, False, [], agent.velocity_history

    # The previous velocity's outcome is this round's score (agent.last_dice, the skill
    # that velocity produced). Filled in here from real scores, never by the LLM.
    history = list(agent.velocity_history)
    if history and not history[-1].outcome_recorded:
        history[-1] = history[-1].model_copy(
            update={"dice_after": agent.last_dice, "outcome_recorded": True}
        )

    reflection = reflect(llm, agent.skill_md, agent.last_code, agent.last_dice, agent.last_iou, neighbourhood, enriched_reflection)
    docs = []
    if enriched_reflection:
        draft, query = extract_grounding_query(reflection)
        docs = retrieve(query, settings)
        grounding = format_for_prompt(docs, settings)
        reflection = enrich_reflection(llm, draft, grounding)
    retrieved_paper_ids = [d.metadata.get("paper_id", "unknown") for d in docs]

    # Velocity update
    window = settings.velocity_history_len
    shown = history[-window:] if window > 0 else []
    best_this_round = _best_score([agent.last_dice] + [p.last_dice for p in neighbourhood])
    v = velocity_update(
        llm, agent.velocity, reflection, agent.skill_md, agent.p_best_skill, g_best_skill,
        shown, agent.last_dice, agent.p_best_score, g_best_score, best_this_round,
    )

    # Skill update
    s = skill_update(llm, agent.skill_md, v)
    if window > 0:
        next_round = history[-1].round + 1 if history else 1
        history.append(VelocityRecord(round=next_round, velocity=v, dice_before=agent.last_dice))
        history = history[-window:]
    return s, v, True, retrieved_paper_ids, history
