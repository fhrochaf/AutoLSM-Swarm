"""LangGraph state for the conversational swarm (orchestration/graph.py).

Each agent carries its pipeline.py, its personal-best pipeline and a short private notebook;
the swarm carries the global-best pipeline, the OBSERVER's report on the last round, and the
blackboard -- the shared list of messages the agents post to while they discuss their next
strategies. The whole state is checkpointed by LangGraph's SqliteSaver after every node, so a
run can be resumed from wherever it stopped, mid-conversation included.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# p_best_score/g_best_score default to float("-inf") (no best yet). Plain JSON has no
# infinity literal, so pydantic's default JSON mode would serialize -inf as null and
# then fail to parse it back as a float -- "constants" emits/reads -Infinity instead.
_INF_SAFE = ConfigDict(ser_json_inf_nan="constants")


class AgentState(BaseModel):
    model_config = _INF_SAFE

    agent_idx: int
    model: str = ""  # "<provider>/<model>" this agent runs on (config.py: agents)
    # Paper ids this agent was seeded with in round 0 (corpus/retrieve.py::retrieve_diverse).
    seed_paper_ids: list[str] = []
    # The agent's last *working* pipeline.py (or, if it has never run successfully, the
    # latest broken attempt).
    pipeline_code: str = ""
    p_best_code: str = ""
    p_best_score: float = float("-inf")
    last_dice: float | None = None  # mean Dice over the evaluation seeds -- the score selection uses
    last_dice_per_seed: dict[int, float] = {}
    last_dice_std: float | None = None
    last_iou: float | None = None
    last_hparams: dict = {}  # HPARAMS of the latest successful run
    last_error: str | None = None  # why the latest run failed; None if it succeeded
    # Private notes the agent writes along with each pipeline and reads back next round --
    # its only memory across rounds besides its pipelines.
    notebook: str = ""


class Post(BaseModel):
    """One message on the blackboard."""

    round: int
    cycle: int
    author: str  # "agent_<i>" or "checker"
    kind: Literal["message", "check_request", "check_result"] = "message"
    text: str


class PendingCheck(BaseModel):
    requester: int  # agent_idx
    question: str


class SwarmState(BaseModel):
    model_config = _INF_SAFE

    run_id: str
    round: int = 0  # the round being worked on (target round count: settings.n_rounds, kept in <run>/run_config.json)
    data_npz_path: str
    agents: list[AgentState]
    g_best_code: str | None = None
    g_best_score: float = float("-inf")
    g_best_agent: int | None = None
    rounds_without_improvement: int = 0  # consecutive rounds g_best_score hasn't improved
    stop_reason: str | None = None

    # The OBSERVER's output for the round that just ran (shared context of the conversation).
    observer_report: str = ""
    observer_tables: str = ""

    # The conversation of the current round. Cleared when the next round's builds start;
    # every round's board is also saved to round_<t>/blackboard.{jsonl,md}.
    blackboard: list[Post] = []
    turn_order: list[int] = []  # speaking order this round (rotated every round)
    cycle: int = 0
    speaker_ptr: int = 0  # index into turn_order of the next agent to speak
    passes_this_cycle: int = 0
    pending_check: PendingCheck | None = None
    checks_used: int = 0  # check requests granted this round
    # Ledger stats of the current round's conversation, per agent: posts, passes, tool calls, ...
    talk_stats: dict[int, dict] = {}


# Classes the checkpointer may deserialize (langgraph's msgpack allowlist).
CHECKPOINT_TYPES = [
    (__name__, cls.__name__) for cls in (AgentState, Post, PendingCheck, SwarmState)
]
