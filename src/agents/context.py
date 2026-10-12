"""What one swarm agent is shown at each step, rebuilt fresh from the swarm state every time.

Two messages plus the phase instruction (the system prompt is the agent's own, set in
agents/swarm_agent.py):

    SHARED   visible to every agent: last round's scores, the OBSERVER's report and tables,
             the global-best pipeline, and the blackboard
    PRIVATE  only this agent: its own current pipeline (score or error), its personal-best
             pipeline and its notebook

In round 0 nothing has run yet: the agent gets only the papers it was seeded with.

Phases: "build_initial" (round 0), "build" (next pipeline, after the conversation), "talk"
(one turn on the blackboard).
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage

from agents.prompts import render
from agents.text_utils import cap_text
from config import Settings
from orchestration.blackboard import format_board
from orchestration.state import AgentState, SwarmState

Phase = Literal["build_initial", "build", "talk"]


def _score(value: float | None) -> str:
    if value is None:
        return "failed"
    if value in (float("inf"), float("-inf")):
        return "none yet"
    return f"{value:.4f}"


def _code_block(code: str, settings: Settings) -> str:
    code = cap_text(code, settings.full_text_char_cap, "\n# ...[truncated]")
    return f"```python\n{code}\n```"


def scores_table(state: SwarmState) -> str:
    rows = ["| agent | model | Dice (mean over seeds) | ± std | personal best | status |", "|---|---|---|---|---|---|"]
    for a in state.agents:
        std = f"{a.last_dice_std:.4f}" if a.last_dice_std is not None else "-"
        status = "ok" if a.last_error is None else f"failed: {cap_text(a.last_error, 120, '...')}"
        rows.append(f"| agent_{a.agent_idx} | {a.model or '-'} | {_score(a.last_dice)} | {std} | {_score(a.p_best_score)} | {status} |")
    return "\n".join(rows)


def shared_context(state: SwarmState, settings: Settings, shown_round: int) -> str:
    parts = [f"# SHARED CONTEXT (every agent sees this)\n\n## Results of round {shown_round}\n{scores_table(state)}"]
    if state.observer_report or state.observer_tables:
        parts.append(f"## OBSERVER's report\n{state.observer_report}\n\n### Metric tables\n{state.observer_tables}")
    if state.g_best_code:
        parts.append(
            f"## Global-best pipeline.py so far (agent_{state.g_best_agent}, Dice={_score(state.g_best_score)})\n"
            + _code_block(state.g_best_code, settings)
        )
    parts.append(f"## Blackboard (conversation of round {shown_round})\n{format_board(state.blackboard)}")
    return "\n\n".join(parts)


def private_context(agent: AgentState, settings: Settings) -> str:
    parts = [f"# PRIVATE CONTEXT (only you, agent_{agent.agent_idx}, see this)"]
    if agent.pipeline_code:
        status = (
            f"Dice={_score(agent.last_dice)}" if agent.last_error is None
            else f"FAILED: {cap_text(agent.last_error, 2000)}"
        )
        parts.append(f"## Your current pipeline.py ({status})\n" + _code_block(agent.pipeline_code, settings))
        if agent.last_hparams:
            parts.append(f"Its HPARAMS: {agent.last_hparams}")
    if agent.p_best_code:
        if agent.p_best_code == agent.pipeline_code:
            parts.append(f"## Your personal-best pipeline.py (Dice={_score(agent.p_best_score)})\nThe same as your current pipeline.")
        else:
            parts.append(
                f"## Your personal-best pipeline.py (Dice={_score(agent.p_best_score)})\n"
                + _code_block(agent.p_best_code, settings)
            )
    parts.append(f"## Your notebook\n{agent.notebook or '(empty)'}")
    return "\n\n".join(parts)


def build_messages(
    state: SwarmState, agent_idx: int, phase: Phase, settings: Settings, seed_papers: str | None = None
) -> list[HumanMessage]:
    agent = state.agents[agent_idx]
    if phase == "build_initial":
        instruction = (
            render("agent.build_initial_user", seed_papers=seed_papers or "(none)")
            if settings.use_rag else render("agent.build_initial_user_no_rag")
        )
        # Round 0: nothing has run yet, so no shared context -- only the instruction.
        return [HumanMessage(instruction)]
    # While the next round's pipelines are written, state.round has already advanced: the
    # results and conversation shown are those of the round before.
    shown_round = state.round - 1 if phase == "build" else state.round
    if phase == "build":
        instruction = render("agent.build_user")
    else:
        instruction = render(
            "conversation.turn_user",
            cycle=state.cycle + 1,
            n_cycles=settings.conversation_cycles,
            checks_left=max(0, settings.max_checks_per_round - state.checks_used),
        )
    return [
        HumanMessage(shared_context(state, settings, shown_round)),
        HumanMessage(private_context(agent, settings) + "\n\n# YOUR TASK NOW\n" + instruction),
    ]
