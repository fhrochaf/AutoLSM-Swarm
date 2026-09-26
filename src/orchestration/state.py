"""LangGraph state for the round loop. Each agent carries a skill (position),
a velocity (semantic update direction), and
its personal-best; the swarm carries a global-best."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# p_best_score/g_best_score default to float("-inf") (no best yet). Plain JSON has no
# infinity literal, so pydantic's default JSON mode would serialize -inf as null and
# then fail to parse it back as a float -- "constants" emits/reads -Infinity instead,
# which round-trips correctly. This matters because resuming a run
# (orchestration/graph.py::load_run_state) reconstructs this model from JSON.
_INF_SAFE = ConfigDict(ser_json_inf_nan="constants")


class VelocityRecord(BaseModel):
    """One past velocity and what it did: the skill it was applied to scored
    dice_before; once the resulting skill has run, dice_after holds that score (None
    with outcome_recorded=True means that run failed)."""

    model_config = _INF_SAFE

    round: int
    velocity: str
    dice_before: float | None = None
    dice_after: float | None = None
    outcome_recorded: bool = False


class AgentState(BaseModel):
    model_config = _INF_SAFE

    agent_idx: int
    skill_md: str = ""
    velocity: str = ""  # semantic velocity direction; "" until the first peer-review update
    velocity_history: list[VelocityRecord] = []  # most recent last, capped by settings.velocity_history_len
    p_best_skill: str = ""
    p_best_score: float = float("-inf")
    last_dice: float | None = None
    last_iou: float | None = None
    last_code: str = ""  # pipeline.py that produced last_dice/last_iou; "" if it failed
    last_skill_changed: bool = True


class SwarmState(BaseModel):
    model_config = _INF_SAFE

    run_id: str
    round: int = 0
    n_rounds: int
    data_npz_path: str
    agents: list[AgentState]
    g_best_skill: str | None = None
    g_best_score: float = float("-inf")
    rounds_without_improvement: int = 0  # consecutive rounds g_best_score hasn't improved
    in_progress_agents: dict[int, AgentState] = {}
