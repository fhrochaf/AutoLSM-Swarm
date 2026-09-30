"""Deterministic, text-ready behavioral comparison of every agent's validation-set
predictions -- the facts behind the peer observation O_i (AgentPSO, Hwang et al., 2026).

Nothing here touches an LLM or agent code: it only sees each agent's predicted binary
masks and the ground truth, on the same tiles, so every number is comparable across
agents. orchestration/peer_review.py feeds the formatted tables to the peer-review LLM
(call A); its report is what every agent reflects on (call B). An agent's peers are
described by *how their pipelines behave*, never by their code.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from eval.metrics import dice, iou

# A positive tile whose best agent still scores below this counts as a "hard" tile.
HARD_TILE_DICE = 0.1
# "Near" a landslide, in pixels -- separates boundary spill from spurious detections.
BOUNDARY_PX = 3

_STRUCT_8 = np.ones((3, 3), dtype=bool)
_STRUCT_HW = np.ones((1, 3, 3), dtype=bool)  # dilate within each tile, never across tiles


def _div(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def _tile_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-tile Dice, shape (N,). Both masks empty counts as a perfect 1.0."""
    inter = np.logical_and(pred, gt).sum(axis=(1, 2)).astype(float)
    denom = pred.sum(axis=(1, 2)) + gt.sum(axis=(1, 2))
    return np.where(denom == 0, 1.0, 2.0 * inter / np.maximum(denom, 1))


def size_edges(gt: np.ndarray) -> list[float]:
    """Tercile edges of the landslide area (px) across positive tiles: the small /
    medium / large bins every agent is stratified by (same edges for all agents)."""
    areas = gt.astype(bool).sum(axis=(1, 2))
    areas = areas[areas > 0]
    if len(areas) < 3:
        return []
    return [float(e) for e in np.percentile(areas, [100 / 3, 200 / 3])]


def agent_metrics(pred: np.ndarray, gt: np.ndarray, edges: list[float]) -> dict:
    """One agent's behavior on (N,H,W) binary masks. Plain floats/ints/None only."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    n = len(gt)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    gt_any = gt.reshape(n, -1).any(axis=1)
    pred_any = pred.reshape(n, -1).any(axis=1)
    fp_px_on_empty = int(pred[~gt_any].sum())

    near_gt = ndimage.binary_dilation(gt, structure=_STRUCT_HW, iterations=BOUNDARY_PX)
    fp_near_gt = int(np.logical_and(pred & ~gt, near_gt).sum())

    # Object-level view (8-connected blobs): fragmentation and missed/spurious objects.
    gt_objs = gt_hit = pred_objs = pred_hit = 0
    gt_areas: list[int] = []
    pred_areas: list[int] = []
    for i in range(n):
        g_lab, g_n = ndimage.label(gt[i], structure=_STRUCT_8)
        p_lab, p_n = ndimage.label(pred[i], structure=_STRUCT_8)
        gt_objs += g_n
        pred_objs += p_n
        if g_n:
            gt_areas += np.bincount(g_lab.ravel())[1:].tolist()
            gt_hit += int(np.sum(ndimage.maximum(pred[i], g_lab, index=np.arange(1, g_n + 1))))
        if p_n:
            pred_areas += np.bincount(p_lab.ravel())[1:].tolist()
            pred_hit += int(np.sum(ndimage.maximum(gt[i], p_lab, index=np.arange(1, p_n + 1))))

    tile_dice = _tile_dice(pred, gt)
    by_size: dict[str, dict] = {}
    if edges:
        area = gt.sum(axis=(1, 2))
        bin_idx = np.digitize(area, edges)
        for b, name in enumerate(("small", "medium", "large")):
            sel = (area > 0) & (bin_idx == b)
            if sel.any():
                g_px = int(gt[sel].sum())
                by_size[name] = {
                    "n_tiles": int(sel.sum()),
                    "tile_dice": float(tile_dice[sel].mean()),
                    "pixel_recall": _div(np.logical_and(pred[sel], gt[sel]).sum(), g_px),
                }

    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "pred_pos_rate": float(pred.mean()),
        "true_pos_rate": float(gt.mean()),
        "tile_precision": _div(int((pred_any & gt_any).sum()), int(pred_any.sum())),
        "tile_recall": _div(int((pred_any & gt_any).sum()), int(gt_any.sum())),
        "fp_tiles": int((pred_any & ~gt_any).sum()),
        "fp_px_on_empty_tiles_frac": _div(fp_px_on_empty, fp),
        "fp_near_gt_frac": _div(fp_near_gt, fp),
        "object_recall": _div(gt_hit, gt_objs),
        "object_precision": _div(pred_hit, pred_objs),
        "pred_blobs_per_tile": pred_objs / n,
        "gt_blobs_per_tile": gt_objs / n,
        "median_pred_blob_px": float(np.median(pred_areas)) if pred_areas else None,
        "median_gt_blob_px": float(np.median(gt_areas)) if gt_areas else None,
        "tile_dice_on_positive_tiles": float(tile_dice[gt_any].mean()) if gt_any.any() else None,
        "by_size": by_size,
    }


def compare_agents(preds: dict[int, np.ndarray], gt: np.ndarray) -> dict:
    """Behavioral comparison of every agent that produced predictions this round.

    Returns {"agents": {idx: metrics}, "swarm": {...}}.
    """
    gt = gt.astype(bool)
    n = len(gt)
    ids = sorted(preds)
    edges = size_edges(gt)
    gt_any = gt.reshape(n, -1).any(axis=1)

    agents = {i: agent_metrics(preds[i], gt, edges) for i in ids}
    tile_d = {i: _tile_dice(preds[i].astype(bool), gt) for i in ids}
    pos = np.where(gt_any)[0]

    swarm: dict = {
        "n_agents": len(ids),
        "n_tiles": n,
        "n_positive_tiles": int(gt_any.sum()),
        "size_edges_px": edges,
    }

    if len(ids) >= 2 and len(pos):
        mat = np.stack([tile_d[i] for i in ids])  # (A,N)
        best = mat.max(axis=0)
        swarm["oracle_tile_dice_on_positive_tiles"] = float(best[pos].mean())
        swarm["hard_tiles"] = int((best[pos] < HARD_TILE_DICE).sum())
        votes = np.stack([preds[i].astype(bool) for i in ids]).sum(axis=0)
        swarm["majority_vote_dice"] = dice(votes * 2 > len(ids), gt)
        for a, i in enumerate(ids):
            agents[i]["wins_on_positive_tiles"] = int(
                np.sum(np.isclose(mat[a, pos], best[pos]) & (best[pos] > 0))
            )
            others = [dice(preds[i].astype(bool), preds[j].astype(bool)) for j in ids if j != i]
            agents[i]["agreement_with_peers"] = float(np.mean(others))
    elif len(pos):
        swarm["hard_tiles"] = int((tile_d[ids[0]][pos] < HARD_TILE_DICE).sum()) if ids else 0

    return {"agents": agents, "swarm": swarm}


# --------------------------------------------------------------------------------------
# Text formatting -- the only form the LLMs ever see.
# --------------------------------------------------------------------------------------


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _seed_cell(dices: list[float] | None) -> str:
    if not dices:
        return "n/a"
    return f"{np.mean(dices):.3f} ± {np.std(dices):.3f} ({', '.join(f'{d:.3f}' for d in dices)})"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def format_comparison(
    cmp: dict,
    failed: dict[int, str] | None = None,
    seed_dice: dict[int, list[float]] | None = None,
    table_seed: int | None = None,
) -> str:
    """The swarm-wide comparison as markdown tables (every agent, same tiles).

    The behavioral tables are computed on one seed's predictions (table_seed);
    seed_dice ({agent: per-seed Dice}) adds the score over every evaluation seed."""
    agents, swarm = cmp["agents"], cmp["swarm"]
    ids = sorted(agents)
    out = [
        f"Validation set: {swarm['n_tiles']} tiles, {swarm['n_positive_tiles']} contain a "
        f"landslide. All agents were scored on exactly these tiles."
    ]
    multi = bool(seed_dice) and any(len(v) > 1 for v in seed_dice.values())
    if multi:
        out.append(
            f"Each pipeline was trained and scored under {max(len(v) for v in seed_dice.values())} seeds. "
            f"The behavioral tables below use the predictions of seed {table_seed}; the last column of "
            "the first table gives the Dice over all seeds."
        )
    if not ids:
        out.append("No agent produced predictions this round.")
    else:
        out.append("\n### Pixel level")
        out.append(
            _table(
                ["agent", "Dice", "IoU", "precision", "recall", "predicted-positive rate / true rate"]
                + (["Dice over seeds: mean ± sd (per seed)"] if multi else []),
                [
                    [
                        str(i),
                        _f(agents[i]["dice"]),
                        _f(agents[i]["iou"]),
                        _f(agents[i]["precision"]),
                        _f(agents[i]["recall"]),
                        f"{agents[i]['pred_pos_rate']:.4f} / {agents[i]['true_pos_rate']:.4f}",
                    ]
                    + ([_seed_cell(seed_dice.get(i))] if multi else [])
                    for i in ids
                ],
            )
        )
        out.append("\n### Tile and object level")
        out.append(
            _table(
                [
                    "agent", "tile precision", "tile recall", "false-alarm tiles",
                    "FP px on empty tiles", "FP px within 3px of a landslide",
                    "object recall", "object precision",
                    "blobs/tile pred vs GT", "median blob px pred vs GT",
                ],
                [
                    [
                        str(i),
                        _f(a["tile_precision"]),
                        _f(a["tile_recall"]),
                        str(a["fp_tiles"]),
                        _pct(a["fp_px_on_empty_tiles_frac"]),
                        _pct(a["fp_near_gt_frac"]),
                        _f(a["object_recall"]),
                        _f(a["object_precision"]),
                        f"{a['pred_blobs_per_tile']:.2f} vs {a['gt_blobs_per_tile']:.2f}",
                        f"{_f(a['median_pred_blob_px'], 0)} vs {_f(a['median_gt_blob_px'], 0)}",
                    ]
                    for i, a in ((i, agents[i]) for i in ids)
                ],
            )
        )
        if swarm.get("size_edges_px"):
            e = swarm["size_edges_px"]
            out.append(
                f"\n### By landslide size (per-tile area terciles: small <= {e[0]:.0f}px, "
                f"medium <= {e[1]:.0f}px, large above) -- tile Dice / pixel recall"
            )
            rows = []
            for i in ids:
                row = [str(i)]
                for name in ("small", "medium", "large"):
                    s = agents[i]["by_size"].get(name)
                    row.append("n/a" if not s else f"{s['tile_dice']:.3f} / {_f(s['pixel_recall'])}")
                rows.append(row)
            out.append(_table(["agent", "small", "medium", "large"], rows))
        if swarm["n_agents"] >= 2:
            out.append("\n### Across agents")
            out.append(
                _table(
                    ["agent", "mean tile Dice (positive tiles)", "tiles where it is the best agent",
                     "mean Dice between its mask and each peer's"],
                    [
                        [
                            str(i),
                            _f(agents[i]["tile_dice_on_positive_tiles"]),
                            str(agents[i].get("wins_on_positive_tiles", "n/a")),
                            _f(agents[i].get("agreement_with_peers")),
                        ]
                        for i in ids
                    ],
                )
            )
            out.append(
                f"Oracle (best agent per positive tile) mean tile Dice: "
                f"{_f(swarm.get('oracle_tile_dice_on_positive_tiles'))}. "
                f"Majority-vote ensemble Dice: {_f(swarm.get('majority_vote_dice'))}. "
                f"Positive tiles where every agent scores under {HARD_TILE_DICE}: "
                f"{swarm.get('hard_tiles', 'n/a')} of {swarm['n_positive_tiles']}."
            )
    if failed:
        out.append(
            "\nAgents with no predictions this round (their run failed): "
            + "; ".join(f"agent {i}: {msg[:160]}" for i, msg in sorted(failed.items()))
        )
    return "\n".join(out)
