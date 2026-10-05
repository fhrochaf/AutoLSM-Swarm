"""Peer review / PSO-style pipeline update, adapted from AgentPSO (Hwang et al., 2026).

Same loop as AgentPSO, with one change of particle: an agent's position is its
pipeline.py rather than a natural-language skill, so the score is optimized on the code
itself. Topology: global-best -- every agent's neighbourhood is the whole rest of the
swarm, and every agent is steered toward one swarm-wide global-best pipeline.

Each round, from round 1 on (all prompts live in src/prompts.yaml):

    A. peer review   O^t     = PeerReview(metrics of every agent on the same validation
                               tiles)  -- one call per round, shared by the whole swarm
    B. reflect       d_i^t   = Reflect(pipeline_i, O^t)
                   (optionally) d_i^t = GroundReflection(d_i^t, retrieved_lit_i)
    C. velocity      v_i^t+1 = VelocityUpdate(v_i^t, d_i^t, pipeline_i, p_best_i, g_best)
    D. position      pipeline_i^t+1 = ApplyVelocity(pipeline_i, v_i^t+1)  (agents/codegen.py)
       (optional) fidelity gate: a judge checks pipeline_i^t+1 against v_i^t+1 and sends a
                  NOT_FAITHFUL update back to the code LLM (agents/fidelity.py,
                  config.fidelity_check)

The peer observation O^t is behavioral and all text: the per-agent metric tables computed
by eval/observe.py from every agent's validation predictions, plus the peer-review LLM's
comparative report on them. It deliberately excludes peers' code (as AgentPSO excludes
peers' skills); the only foreign code an agent sees is the personal-best and global-best
pipelines, in the velocity update. Reflect sees nothing but the agent's own pipeline and
this round's peer review.

When the agent is chosen for an enriched reflection (config.py: enriched_reflection_all /
random_enriched_reflection / the current global-best holder), Reflect also closes with one
explicit "GROUNDING QUERY: ..." line that is used, unmodified, as the corpus search query,
and GroundReflection revises the draft reflection against the retrieved papers before it
reaches VelocityUpdate.

With no peers (n_agents == 1) the peer review simply describes the one agent.
"""
from __future__ import annotations

import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_core.language_models import BaseChatModel

from agents import artifacts
from agents.codegen import apply_velocity
from agents.fidelity import FidelityOutcome, enforce_fidelity
from agents.prompts import DATASET_DESCRIPTION, render
from agents.structured import VelocityAnswer, invoke_structured
from config import Settings
from corpus.retrieve import format_for_prompt, retrieve
from eval import observe
from orchestration.state import AgentState, SwarmState, VelocityRecord

settings = Settings()

_MAX_CODE_CHARS = settings.full_text_char_cap

# Paper ids are Scopus EIDs, e.g. "2-s2.0-105000159569" -- the same id scheme used as
# both the corpus JSON filenames and the "[paper_id]" citation style the prompts
# already use throughout (src/prompts.yaml, corpus/retrieve.py::format_for_prompt).
_CITATION_RE = re.compile(r"2-s2\.0-\d+")

# Matches the trailing "GROUNDING QUERY: ..." line Reflect is asked to emit.
_GROUNDING_QUERY_RE = re.compile(r"(?im)^GROUNDING QUERY:\s*(.+?)\s*$")



def extract_cited_papers(text: str) -> list[str]:
    """Paper ids actually cited in a piece of generated text (e.g. a pipeline.py's
    comments), in first-seen order, deduplicated. This is the real citation trail for a
    report -- distinct from which papers were merely retrieved/shown to the agent that
    round."""
    seen: list[str] = []
    for match in _CITATION_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def _format_score(value: float | None) -> str:
    if value is None:
        return "N/A (run failed)"
    if value in (float("inf"), float("-inf")):
        return "N/A (no best yet)"
    return f"{value:.4f}"


def _best_score(dices: list[float | None]) -> float | None:
    known = [d for d in dices if d is not None]
    return max(known) if known else None


def _cap(code: str) -> str:
    if len(code) > _MAX_CODE_CHARS:
        return code[:_MAX_CODE_CHARS] + "\n# ...[truncated]"
    return code


def count_changed_lines(old: str, new: str) -> int:
    """Lines added plus removed between two versions of a pipeline.py -- how big the
    move was, logged per round so the size of each update (a one-line tweak, or a larger
    multi-change one) can be audited."""
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
    return sum(
        1 for line in diff if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )


def changed_hparams(old: dict, new: dict) -> list[str]:
    """Keys of HPARAMS that were added, removed, or changed value."""
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


# --------------------------------------------------------------------------------------
# A. Peer review -- the peer observation, shared by the whole swarm for one round.
# --------------------------------------------------------------------------------------


@dataclass
class PeerReview:
    comparison: dict  # eval.observe.compare_agents output
    table_text: str  # exact, code-computed metric tables
    report: str  # peer-review LLM's comparative report on those tables
    agent_metrics: dict[int, dict] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)  # each agent's score (mean Dice over seeds)

    def text(self) -> str:
        return f"{self.table_text}\n\n### Coordinator's comparative report\n{self.report}"


def run_peer_review(
    peer_review_llm: BaseChatModel, state: SwarmState, settings: Settings, runs_dir: Path
) -> PeerReview:
    """Call A. Reads every agent's validation predictions from the round that just
    finished (round state.round - 1), computes the behavioral tables deterministically,
    and has the peer-review LLM write the comparative report on them. The report is
    cached next to the round it describes, so a resumed round does not pay for it twice.

    Behavior is read from the first evaluation seed's predictions; each agent's Dice over
    every seed is added to the tables."""
    prev_round = state.round - 1
    y_val = np.load(state.data_npz_path)["y_val"]
    table_seed = settings.active_seeds()[0]

    preds: dict[int, np.ndarray] = {}
    failed: dict[int, str] = {}
    seed_dice: dict[int, list[float]] = {}
    for agent in state.agents:
        path = artifacts.val_preds_path(
            artifacts.agent_round_dir(runs_dir, state.run_id, prev_round, agent.agent_idx),
            table_seed,
        )
        if agent.last_dice is not None and path.exists():
            preds[agent.agent_idx] = np.load(path)["preds"]
            seed_dice[agent.agent_idx] = [d for _, d in sorted(agent.last_dice_per_seed.items())]
        else:
            failed[agent.agent_idx] = agent.last_error or "run failed"

    if preds:
        comparison = observe.compare_agents(preds, y_val)
    else:
        comparison = {
            "agents": {},
            "swarm": {
                "n_tiles": len(y_val),
                "n_positive_tiles": int(y_val.reshape(len(y_val), -1).any(axis=1).sum()),
            },
        }
    table_text = observe.format_comparison(comparison, failed, seed_dice, table_seed)

    round_path = artifacts.round_dir(runs_dir, state.run_id, prev_round)
    report_path = round_path / "peer_review.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
    elif not preds:
        report = "(no agent produced predictions, so there is nothing to compare)"
    else:
        response = peer_review_llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
            [
                ("system", render("peer_review.system")),
                ("human", render("peer_review.user", dataset_description=DATASET_DESCRIPTION, tables=table_text)),
            ]
        )
        report = response.text.strip()
    report_path.write_text(report, encoding="utf-8")
    (round_path / "peer_review_tables.md").write_text(table_text, encoding="utf-8")

    scores = {a.agent_idx: a.last_dice for a in state.agents if a.agent_idx in preds}
    return PeerReview(comparison, table_text, report, comparison["agents"], scores)


# --------------------------------------------------------------------------------------
# B. Reflect (+ optional GroundReflection)
# --------------------------------------------------------------------------------------


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
    agent: AgentState,
    review: PeerReview,
    enriched_reflection: bool = False,
) -> str:
    """Reflect(pipeline_i, O^t): the agent's own pipeline and this round's peer review
    are its only inputs."""
    prompt = render(
        "reflect.user",
        dataset_description=DATASET_DESCRIPTION,
        agent_idx=agent.agent_idx,
        pipeline_code=_cap(agent.pipeline_code),
        peer_review=review.text(),
    )
    system = render("reflect.system")
    if enriched_reflection:
        system = f"{system}\n{render('reflect.grounding_suffix')}"

    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", system), ("human", prompt)]
    )
    return response.text.strip()


def enrich_reflection(llm: BaseChatModel, reflection: str, grounding: str) -> str:
    prompt = render("reflect.enrich_user", reflection=reflection, grounding=grounding)
    response = llm.with_retry(stop_after_attempt=settings.llm_retry_attempts).invoke(
        [("system", render("reflect.enrich_system")), ("human", prompt)]
    )
    return response.text.strip()


# --------------------------------------------------------------------------------------
# C. Velocity update
# --------------------------------------------------------------------------------------


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
        lines.append(f"- Round {rec.round}: applied to a pipeline scoring Dice={before} -> {outcome}\n  {rec.velocity}")
    return "\n".join(lines)


def velocity_update(
    llm: BaseChatModel,
    prev_velocity: str,
    reflection: str,
    pipeline_code: str,
    p_best_code: str,
    g_best_code: str | None,
    history: list[VelocityRecord] | None = None,
    current_dice: float | None = None,
    p_best_score: float | None = None,
    g_best_score: float | None = None,
    best_this_round: float | None = None,
) -> tuple[str, str]:
    """Call C. Returns (velocity, reasoning). The answer is always requested as JSON with the
    fields `reasoning` and `final_velocity` (agents/structured.py), whatever the prompt says, and
    only `final_velocity` is used as the velocity: it is what the code LLM, the fidelity judge and
    the trajectory see. If no JSON can be obtained at all, the raw reply is used as the velocity."""
    fence = "```"
    p_best_block = (
        f"{fence}python\n{_cap(p_best_code)}\n{fence}"
        if p_best_code and p_best_code != pipeline_code
        else "(same as current -- no better round yet)"
    )
    g_best_block = (
        f"{fence}python\n{_cap(g_best_code)}\n{fence}"
        if g_best_code and g_best_code != pipeline_code
        else "(same as current -- you hold the global best, or none is recorded yet)"
    )
    prompt = render(
        "velocity.user",
        prev_velocity=prev_velocity or "(none yet -- this is the first update)",
        trajectory=_format_trajectory(history or []),
        current_dice=_format_score(current_dice),
        best_this_round=_format_score(best_this_round),
        reflection=reflection,
        pipeline_code=_cap(pipeline_code),
        p_best_score=_format_score(p_best_score),
        p_best_block=p_best_block,
        g_best_score=_format_score(g_best_score),
        g_best_block=g_best_block,
    )
    parsed, raw = invoke_structured(llm, VelocityAnswer, [("system", render("velocity.system")), ("human", prompt)])
    if parsed is not None and parsed.final_velocity.strip():
        return parsed.final_velocity.strip(), parsed.reasoning.strip()
    print("[velocity] no usable JSON answer from the LLM; using its raw reply as the velocity", file=sys.stderr)
    return raw.strip(), ""


# --------------------------------------------------------------------------------------
# B + C + D for one agent
# --------------------------------------------------------------------------------------


@dataclass
class PipelineUpdate:
    code: str
    velocity: str
    reflection: str
    velocity_history: list[VelocityRecord]
    retrieved_papers: list[str]
    changed_lines: int
    fidelity: FidelityOutcome | None = None  # None when the gate is off
    velocity_reasoning: str = ""  # the `reasoning` field of the velocity answer ("" if the LLM gave none)


def reflect_and_update(
    llm: BaseChatModel,
    code_llm: BaseChatModel,
    agent: AgentState,
    review: PeerReview,
    g_best_code: str | None,
    g_best_score: float,
    settings: Settings,
    round_idx: int,
    enriched_reflection: bool = False,
    judge_llm: BaseChatModel | None = None,
) -> PipelineUpdate:
    """Run Reflect (-> GroundReflection) -> VelocityUpdate -> ApplyVelocity (-> fidelity
    gate, when settings.fidelity_check) for one agent against this round's peer review."""
    # The previous velocity's outcome is this round's score (agent.last_dice, the
    # pipeline that velocity produced). Filled in here from real scores, never by the LLM.
    history = list(agent.velocity_history)
    if history and not history[-1].outcome_recorded:
        history[-1] = history[-1].model_copy(
            update={"dice_after": agent.last_dice, "outcome_recorded": True}
        )

    enriched_reflection = enriched_reflection and settings.use_rag
    reflection = reflect(llm, agent, review, enriched_reflection)
    docs = []
    if enriched_reflection:
        draft, query = extract_grounding_query(reflection)
        docs = retrieve(query, settings)
        reflection = enrich_reflection(llm, draft, format_for_prompt(docs, settings))
    retrieved_paper_ids = [d.metadata.get("paper_id", "unknown") for d in docs]

    window = settings.velocity_history_len
    shown = history[-window:] if window > 0 else []
    v, velocity_reasoning = velocity_update(
        llm, agent.velocity, reflection, agent.pipeline_code, agent.p_best_code, g_best_code,
        shown, agent.last_dice, agent.p_best_score, g_best_score,
        _best_score(list(review.scores.values())),
    )

    new_code = apply_velocity(code_llm, agent.pipeline_code, v, velocity_reasoning)
    fidelity = None
    if settings.fidelity_check:
        if judge_llm is None:
            raise ValueError("settings.fidelity_check is on but no judge_llm was given to reflect_and_update")
        fidelity = enforce_fidelity(
            judge_llm, code_llm, v, agent.pipeline_code, new_code,
            settings.max_fidelity_iters, settings.fidelity_web_search, velocity_reasoning,
        )
        new_code = fidelity.code
    if window > 0:
        history.append(VelocityRecord(round=round_idx, velocity=v, dice_before=agent.last_dice))
        history = history[-window:]
    return PipelineUpdate(
        new_code, v, reflection, history, retrieved_paper_ids,
        count_changed_lines(agent.pipeline_code, new_code), fidelity, velocity_reasoning,
    )
