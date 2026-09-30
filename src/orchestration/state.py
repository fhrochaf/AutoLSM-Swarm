"""LangGraph state for the round loop. Each agent carries a pipeline.py (position), a
velocity (semantic update direction), and its personal-best pipeline; the swarm carries
a global-best pipeline."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# p_best_score/g_best_score default to float("-inf") (no best yet). Plain JSON has no
# infinity literal, so pydantic's default JSON mode would serialize -inf as null and
# then fail to parse it back as a float -- "constants" emits/reads -Infinity instead,
# which round-trips correctly. This matters because resuming a run
# (orchestration/graph.py::load_run_state) reconstructs this model from JSON.
_INF_SAFE = ConfigDict(ser_json_inf_nan="constants")


class VelocityRecord(BaseModel):
    """One past velocity and what it did: the pipeline it was applied to scored
    dice_before; once the resulting pipeline has run, dice_after holds that score (None
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
    # The agent's position: its last *working* pipeline.py (or, if it has never run
    # successfully, the latest broken attempt -- the only thing to revise).
    pipeline_code: str = ""
    velocity: str = ""  # semantic velocity direction; "" until the first peer-review update
    velocity_history: list[VelocityRecord] = []  # most recent last, capped by settings.velocity_history_len
    p_best_code: str = ""
    p_best_score: float = float("-inf")
    last_dice: float | None = None  # mean Dice over the evaluation seeds -- the score selection uses
    last_dice_per_seed: dict[int, float] = {}
    last_dice_std: float | None = None
    last_iou: float | None = None
    last_hparams: dict = {}  # HPARAMS of the latest successful run
    last_error: str | None = None  # why the latest run failed; None if it succeeded


class SwarmState(BaseModel):
    model_config = _INF_SAFE

    run_id: str
    round: int = 0
    n_rounds: int
    data_npz_path: str
    agents: list[AgentState]
    g_best_code: str | None = None
    g_best_score: float = float("-inf")
    rounds_without_improvement: int = 0  # consecutive rounds g_best_score hasn't improved
    in_progress_agents: dict[int, AgentState] = {}
