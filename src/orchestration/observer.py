"""The OBSERVER: after every agent's pipeline has run, compares their behavior on the same
validation tiles and writes the report every agent reads before the conversation.

The numbers are computed deterministically (eval/observe.py, plus eval/observe_detail.py when
settings.observation_detail == "detailed") from each agent's saved validation predictions of the
first evaluation seed; the OBSERVER agent (role-2 LLM, no tools) only turns those tables into a
comparative report. Both are saved to round_<t>/observer{,_tables}.md and go into the shared
context of every agent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage

from agents import artifacts
from agents.prompts import DATASET_DESCRIPTION, render
from agents.swarm_agent import final_text, invoke_agent, make_llm, transcript
from config import Settings
from data.registry import get_dataset_spec
from eval import observe, observe_detail
from orchestration.state import SwarmState


def _detailed_observation(
    preds: dict[int, np.ndarray], y_val: np.ndarray, state: SwarmState, settings: Settings, table_seed: int,
) -> str:
    """The detailed tile-level section: stratified results, object-level error types, and the
    persistent-difficulty history of every pipeline of this run so far (per-tile stats cached
    next to each pipeline, back-filled from the saved predictions when absent)."""
    run_dir = settings.runs_dir / state.run_id
    descriptors = observe_detail.tile_descriptors_cached(
        run_dir / "_tile_descriptors.npz", Path(state.data_npz_path),
        get_dataset_spec(settings.dataset_name).tile_descriptors,
    )
    entries = []
    for r in range(state.round + 1):
        for a in state.agents:
            d = artifacts.agent_round_dir(settings.runs_dir, state.run_id, r, a.agent_idx)
            entries.append((r, a.agent_idx, artifacts.val_preds_path(d, table_seed), d / "tile_stats.npz"))
    return observe_detail.format_detail(
        preds, y_val, observe_detail.make_strata(y_val, descriptors), observe_detail.object_size_edges(y_val), entries
    )


def observation_tables(state: SwarmState, settings: Settings) -> tuple[str, bool]:
    """The metric tables of the round state.round, and whether any agent produced predictions."""
    y_val = np.load(state.data_npz_path)["y_val"]
    table_seed = settings.active_seeds()[0]

    preds: dict[int, np.ndarray] = {}
    failed: dict[int, str] = {}
    seed_dice: dict[int, list[float]] = {}
    for agent in state.agents:
        path = artifacts.val_preds_path(
            artifacts.agent_round_dir(settings.runs_dir, state.run_id, state.round, agent.agent_idx), table_seed
        )
        if agent.last_error is None and agent.last_dice is not None and path.exists():
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
    tables = observe.format_comparison(comparison, failed, seed_dice, table_seed)
    if settings.observation_detail == "detailed" and preds:
        tables += "\n" + _detailed_observation(preds, y_val, state, settings, table_seed)
    return tables, bool(preds)


def run_observer(state: SwarmState, settings: Settings) -> tuple[str, str]:
    """Returns (report, tables) for the round state.round. The report is cached next to the
    round, so a resumed run does not pay for it twice."""
    tables, any_preds = observation_tables(state, settings)
    round_path = artifacts.round_dir(settings.runs_dir, state.run_id, state.round)
    report_path = round_path / "observer.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
    elif not any_preds:
        report = "(no agent produced predictions this round, so there is nothing to compare)"
    else:
        system = render("observer.system")
        out = invoke_agent(
            make_llm(settings, 2), [], system,
            [HumanMessage(render("observer.user", dataset_description=DATASET_DESCRIPTION, tables=tables))],
            settings, name="observer",
        )
        report = final_text(out)
        (round_path / "observer_transcript.md").write_text(transcript(out, system), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    (round_path / "observer_tables.md").write_text(tables, encoding="utf-8")
    return report, tables
