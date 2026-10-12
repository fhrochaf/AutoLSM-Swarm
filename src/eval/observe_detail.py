"""Detailed tile-level observation, appended to the peer-review tables when
`settings.observation_detail == "detailed"`.

Like observe.py, nothing here touches an LLM or agent code -- only predicted masks, the
ground truth and a few per-tile descriptors of the validation inputs. It answers three
questions the aggregate tables cannot:

  1. Where does each pipeline win or lose? Stratified results: landslide size, number of
     objects, and terrain / vegetation terciles (dataset-provided `tile_descriptors`).
  2. What kind of mistakes? Object-level error types: missed / under-covered / fragmented
     landslide objects (by object size), and spurious or merged predicted blobs.
  3. Which tiles does the whole swarm fail on, round after round? A persistent-difficulty
     summary built from every earlier pipeline of the run (cached per agent-round as
     `tile_stats.npz`, back-filled from the saved predictions when missing).

Tile groups are described by their characteristics only, never by tile id: the agents
should learn from what makes a tile hard, not tune to named validation tiles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from eval.observe import _STRUCT_8, _f, _pct, _table, size_edges

# A positive tile counts as solved when a pipeline's tile Dice reaches this.
SOLVED_DICE = 0.5
# Pipelines whose pooled Dice is below this are collapses (e.g. all-background) and are left
# out of the difficulty history -- they would make every tile look hard.
MIN_POOLED_DICE = 0.3
# Empty tiles flagged as landslide by at least this share of pipelines: persistent false alarms.
PERSISTENT_FP_SHARE = 0.8
INTERMITTENT_FP_SHARE = 0.2
# The difficulty summary needs at least this many non-collapsed pipelines to mean anything.
MIN_PIPELINES = 3
# A landslide object counts as covered when this share of its pixels is predicted.
COVERED = 0.5

LEVELS3 = ["low", "mid", "high"]

TileDescriptorsFn = Callable[[np.ndarray], "dict[str, np.ndarray]"]


# --------------------------------------------------------------------------------------
# Per-tile building blocks
# --------------------------------------------------------------------------------------


def tile_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """Per-tile intersection and predicted pixel counts, shape (N,). With the ground-truth
    pixel counts these give every per-tile Dice / false-alarm flag, and are small enough
    to cache for every pipeline of a run."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return {
        "inter": np.logical_and(pred, gt).sum(axis=(1, 2)).astype(np.int32),
        "pred_px": pred.sum(axis=(1, 2)).astype(np.int32),
    }


def _tile_dice(stats: dict[str, np.ndarray], gt_px: np.ndarray) -> np.ndarray:
    """Per-tile Dice on positive tiles, NaN on empty ones."""
    denom = stats["pred_px"] + gt_px
    return np.where(gt_px > 0, 2.0 * stats["inter"] / np.maximum(denom, 1), np.nan)


def _pooled_dice(stats: dict[str, np.ndarray], gt_px: np.ndarray) -> float:
    denom = float(stats["pred_px"].sum() + gt_px.sum())
    return 2.0 * float(stats["inter"].sum()) / denom if denom else 1.0


def load_or_make_tile_stats(preds_path: Path, cache_path: Path, gt: np.ndarray) -> dict[str, np.ndarray] | None:
    """The cached per-tile stats of one pipeline, computed from its saved predictions on
    first use. None when the pipeline never produced predictions."""
    if cache_path.exists():
        z = np.load(cache_path)
        return {"inter": z["inter"], "pred_px": z["pred_px"]}
    if not preds_path.exists():
        return None
    stats = tile_stats(np.load(preds_path)["preds"], gt)
    np.savez_compressed(cache_path, **stats)
    return stats


def tile_descriptors_cached(cache_path: Path, data_npz_path: Path, fn: TileDescriptorsFn | None) -> dict[str, np.ndarray] | None:
    """Per-tile descriptors of the validation inputs (e.g. mean slope), computed once per
    run from X_val and cached -- X_val is large, so it is not reloaded every round."""
    if fn is None:
        return None
    if cache_path.exists():
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}
    desc = fn(np.load(data_npz_path)["X_val"])
    np.savez(cache_path, **desc)
    return desc


# --------------------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------------------


def make_strata(gt: np.ndarray, descriptors: dict[str, np.ndarray] | None) -> list[tuple[str, list[str], np.ndarray, str]]:
    """[(name, level names, level index per tile, scope)]. Scope "pos" strata only exist on
    landslide tiles (index -1 elsewhere); scope "all" strata (terrain terciles) cover every
    tile, so they also describe where false alarms happen."""
    gt = gt.astype(bool)
    area = gt.sum(axis=(1, 2))
    pos = area > 0
    out: list[tuple[str, list[str], np.ndarray, str]] = []
    edges = size_edges(gt)
    if edges:
        out.append(("size", ["small", "medium", "large"], np.where(pos, np.digitize(area, edges), -1), "pos"))
    n_obj = np.array([ndimage.label(g, structure=_STRUCT_8)[1] for g in gt])
    out.append(("objects", ["1", "2-4", "5+"], np.where(pos, np.digitize(n_obj, [2, 5]), -1), "pos"))
    for name, v in (descriptors or {}).items():
        out.append((name, list(LEVELS3), np.digitize(v, np.percentile(v, [100 / 3, 200 / 3])), "all"))
    return out


# --------------------------------------------------------------------------------------
# Object-level error types
# --------------------------------------------------------------------------------------


def object_size_edges(gt: np.ndarray) -> list[float]:
    """Tercile edges of single landslide *object* areas (px) over the validation set."""
    areas: list[int] = []
    for g in gt.astype(bool):
        lab, n = ndimage.label(g, structure=_STRUCT_8)
        if n:
            areas += np.bincount(lab.ravel())[1:].tolist()
    if len(areas) < 3:
        return []
    return [float(e) for e in np.percentile(areas, [100 / 3, 200 / 3])]


def object_errors(pred: np.ndarray, gt: np.ndarray, obj_edges: list[float]) -> dict:
    """What happens to each ground-truth landslide object and each predicted blob."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    cats = {"missed": 0, "under-covered": 0, "fragmented": 0, "detected": 0}
    by_size = {s: {"n": 0, "lost": 0} for s in ("small", "medium", "large")}
    spurious: list[int] = []
    n_pred_blobs = merged = 0
    for p, g in zip(pred, gt):
        g_lab, g_n = ndimage.label(g, structure=_STRUCT_8)
        p_lab, p_n = ndimage.label(p, structure=_STRUCT_8)
        n_pred_blobs += p_n
        if not g_n and not p_n:
            continue
        both = (g_lab > 0) & (p_lab > 0)
        pairs = np.unique(np.stack([g_lab[both], p_lab[both]], axis=1), axis=0) if both.any() else np.zeros((0, 2), int)
        blobs_per_obj = np.bincount(pairs[:, 0], minlength=g_n + 1)
        objs_per_blob = np.bincount(pairs[:, 1], minlength=p_n + 1)
        if g_n:
            g_area = np.bincount(g_lab.ravel(), minlength=g_n + 1)
            cover = np.bincount(g_lab[both], minlength=g_n + 1)
            for o in range(1, g_n + 1):
                c = cover[o] / g_area[o]
                if cover[o] == 0:
                    k = "missed"
                elif c < COVERED:
                    k = "under-covered"
                elif blobs_per_obj[o] >= 2:
                    k = "fragmented"
                else:
                    k = "detected"
                cats[k] += 1
                if obj_edges:
                    s = ("small", "medium", "large")[int(np.digitize(g_area[o], obj_edges))]
                    by_size[s]["n"] += 1
                    by_size[s]["lost"] += k in ("missed", "under-covered")
        if p_n:
            p_area = np.bincount(p_lab.ravel(), minlength=p_n + 1)
            for b in range(1, p_n + 1):
                if objs_per_blob[b] == 0:
                    spurious.append(int(p_area[b]))
                elif objs_per_blob[b] >= 2:
                    merged += 1
    return {
        "cats": cats, "n_gt_objects": sum(cats.values()), "by_size": by_size, "n_pred_blobs": n_pred_blobs,
        "spurious": len(spurious), "median_spurious_px": float(np.median(spurious)) if spurious else None, "merged": merged,
    }


# --------------------------------------------------------------------------------------
# Persistent difficulty (history over the whole run)
# --------------------------------------------------------------------------------------


def _history(entries: list[tuple[int, int, Path, Path]], gt: np.ndarray, gt_px: np.ndarray) -> dict | None:
    rounds, agents, dice, flag = [], [], [], []
    for r, a, preds_path, cache_path in entries:
        s = load_or_make_tile_stats(preds_path, cache_path, gt)
        if s is None or _pooled_dice(s, gt_px) < MIN_POOLED_DICE:
            continue
        rounds.append(r)
        agents.append(a)
        dice.append(_tile_dice(s, gt_px))
        flag.append(s["pred_px"] > 0)
    if len(rounds) < MIN_PIPELINES:
        return None
    return {"rounds": np.array(rounds), "agents": np.array(agents), "dice": np.stack(dice), "flag": np.stack(flag)}


def _classify(h: dict, gt_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Category per tile. Positive tiles: 0 solved by most pipelines, 1 contested (some
    pipeline solves it, most do not), 2 persistently hard (none ever did); -1 on empty
    tiles. Empty tiles: 0 clean, 1 intermittent false alarm, 2 persistent false alarm;
    -1 on positive tiles."""
    pos = gt_px > 0
    d = h["dice"][:, pos]                      # (pipelines, positive tiles): no NaN here
    mean = np.full(len(pos), np.nan)
    best = np.full(len(pos), np.nan)
    mean[pos], best[pos] = d.mean(axis=0), d.max(axis=0)
    cat_pos = np.where(pos, np.where(mean >= SOLVED_DICE, 0, np.where(best >= SOLVED_DICE, 1, 2)), -1)
    share = h["flag"].mean(axis=0)
    cat_neg = np.where(~pos, np.where(share >= PERSISTENT_FP_SHARE, 2, np.where(share >= INTERMITTENT_FP_SHARE, 1, 0)), -1)
    return cat_pos, cat_neg


POS_CATS = ["solved by most pipelines", "contested (some pipeline solves it, most do not)", "persistently hard (no pipeline has reached Dice 0.5)"]
NEG_CATS = ["clean (flagged by under 20% of pipelines)", "intermittent false alarms (20-80%)", "persistent false alarms (flagged by 80%+ of pipelines)"]


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def _mean(x: np.ndarray) -> float | None:
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else None


def format_detail(
    preds: dict[int, np.ndarray],
    gt: np.ndarray,
    strata: list[tuple[str, list[str], np.ndarray, str]],
    obj_edges: list[float],
    history_entries: list[tuple[int, int, Path, Path]],
) -> str:
    """The detailed observation as markdown: stratified results, object-level error types,
    persistent difficulty. `history_entries` = (round, agent, predictions path, cache path)
    for every earlier agent-round, including the one being reviewed."""
    gt = gt.astype(bool)
    gt_px = gt.sum(axis=(1, 2)).astype(np.int64)
    pos = gt_px > 0
    ids = sorted(preds)
    cur = {i: tile_stats(preds[i], gt) for i in ids}
    cur_dice = {i: _tile_dice(cur[i], gt_px) for i in ids}
    out: list[str] = ["\n## Detailed tile-level observation"]

    # ---- 1. stratified results
    out.append(
        "\n### Where it wins and loses: tile Dice on landslide tiles, by tile type\n"
        "Strata come from the validation tiles themselves: landslide size (area terciles), number of landslide "
        "objects, and terciles of mean slope / elevation / vegetation (NDVI) of the tile's input."
    )
    header, counts = ["agent"], ["tiles in group"]
    cols: list[tuple[np.ndarray, str]] = []
    for name, levels, idx, scope in strata:
        for li, level in enumerate(levels):
            sel = (idx == li) & pos
            header.append(f"{name}: {level}")
            counts.append(str(int(sel.sum())))
            cols.append((sel, name))
    rows = [counts] + [[str(i)] + [_f(_mean(cur_dice[i][sel])) for sel, _ in cols] for i in ids]
    out.append(_table(header, rows))

    terrain = [(n, lv, idx) for n, lv, idx, sc in strata if sc == "all"]
    empty = ~pos
    if terrain and empty.any():
        out.append("\n### Where it raises false alarms: share of landslide-free tiles flagged as landslide, by terrain")
        header, counts = ["agent"], ["landslide-free tiles in group"]
        sels = []
        for name, levels, idx in terrain:
            for li, level in enumerate(levels):
                sel = (idx == li) & empty
                header.append(f"{name}: {level}")
                counts.append(str(int(sel.sum())))
                sels.append(sel)
        rows = [counts] + [
            [str(i)] + [_pct(float((cur[i]["pred_px"][sel] > 0).mean())) if sel.any() else "n/a" for sel in sels] for i in ids
        ]
        out.append(_table(header, rows))

    # ---- 2. object-level error types
    out.append(
        "\n### What kind of mistakes: fate of every landslide object and every predicted blob\n"
        "A landslide object is *missed* (no predicted pixel on it), *under-covered* (under "
        f"{COVERED:.0%} of its pixels predicted), *fragmented* (covered, but by 2+ separate blobs) or *detected*. "
        "A predicted blob is *spurious* if it touches no landslide, *merged* if it touches 2+ landslide objects."
    )
    errs = {i: object_errors(preds[i], gt, obj_edges) for i in ids}
    rows = []
    for i in ids:
        e = errs[i]
        n = max(e["n_gt_objects"], 1)
        row = [str(i)] + [_pct(e["cats"][k] / n) for k in ("missed", "under-covered", "fragmented", "detected")]
        for s in ("small", "medium", "large"):
            b = e["by_size"][s]
            row.append(_pct(b["lost"] / b["n"]) if b["n"] else "n/a")
        row += [str(e["spurious"]), _f(e["median_spurious_px"], 0), str(e["merged"])]
        rows.append(row)
    out.append(
        _table(
            ["agent", "missed", "under-covered", "fragmented", "detected",
             "missed or under-covered: small objects", "medium", "large",
             "spurious blobs", "median spurious blob px", "merged blobs"],
            rows,
        )
    )
    if obj_edges:
        out.append(f"Object size terciles (px): small <= {obj_edges[0]:.0f}, medium <= {obj_edges[1]:.0f}, large above.")

    # ---- 3. persistent difficulty
    h = _history(history_entries, gt, gt_px)
    if h is None:
        out.append(
            f"\n### Persistent difficulty across the run\nNot enough history yet (needs {MIN_PIPELINES}+ "
            f"non-collapsed pipelines, pooled Dice >= {MIN_POOLED_DICE})."
        )
        return "\n".join(out)

    cat_pos, cat_neg = _classify(h, gt_px)
    n_rounds = len(set(h["rounds"].tolist()))
    out.append(
        "\n### Persistent difficulty across the run\n"
        f"Built from {len(h['rounds'])} earlier pipelines over {n_rounds} round(s) (collapsed pipelines with pooled "
        f"Dice < {MIN_POOLED_DICE} excluded). Tiles are grouped by how every pipeline so far has fared on them; the groups "
        "are described by their characteristics, not by tile id."
    )

    def composition(sel: np.ndarray) -> str:
        parts = []
        for name, levels, idx, _ in strata:
            counts = np.array([int(((idx == li) & sel).sum()) for li in range(len(levels))], float)
            if counts.sum():
                parts.append(f"{name} " + "/".join(levels) + " = " + "/".join(f"{100 * c / counts.sum():.0f}%" for c in counts))
        return "; ".join(parts)

    rows = []
    for c, label in enumerate(POS_CATS):
        sel = cat_pos == c
        rows.append([label, str(int(sel.sum())), composition(sel) if sel.any() else "-"])
    out.append("\nLandslide tiles:\n" + _table(["group", "tiles", "make-up of the group (share of its tiles per level)"], rows))
    rows = []
    for c, label in enumerate(NEG_CATS):
        sel = cat_neg == c
        rows.append([label, str(int(sel.sum())), composition(sel) if sel.any() else "-"])
    out.append("\nLandslide-free tiles:\n" + _table(["group", "tiles", "make-up of the group (share of its tiles per level)"], rows))

    rows = []
    for i in ids:
        d, flagged = cur_dice[i], cur[i]["pred_px"] > 0
        rows.append([
            str(i),
            _f(_mean(d[cat_pos == 0])), _f(_mean(d[cat_pos == 1])), _f(_mean(d[cat_pos == 2])),
            f"{int(np.sum((cat_pos == 1) & (np.nan_to_num(d) >= SOLVED_DICE)))} of {int((cat_pos == 1).sum())}",
            _pct(float(flagged[cat_neg == 2].mean())) if (cat_neg == 2).any() else "n/a",
            _pct(float(flagged[cat_neg == 1].mean())) if (cat_neg == 1).any() else "n/a",
        ])
    out.append(
        "\nThis round's pipelines on those groups:\n"
        + _table(
            ["agent", "Dice: solved group", "Dice: contested", "Dice: persistently hard", "contested tiles it solves (Dice >= 0.5)",
             "flags persistent false alarms", "flags intermittent ones"],
            rows,
        )
    )

    # does the swarm learn the hard tiles? mean Dice per round on the contested / hard groups
    rr = sorted(set(h["rounds"].tolist()))
    pick = rr if len(rr) <= 8 else [rr[int(round(k))] for k in np.linspace(0, len(rr) - 1, 8)]
    trend = []
    for c, label in ((1, "contested"), (2, "persistently hard")):
        sel = cat_pos == c
        if not sel.any():
            continue
        vals = []
        for r in pick:
            m = h["rounds"] == r
            vals.append(f"r{r}: {_f(_mean(np.nanmean(h['dice'][m][:, sel], axis=0)))}")
        trend.append(f"- {label} tiles, mean tile Dice of the round's pipelines: " + ", ".join(vals))
    if trend:
        out.append("\nIs the swarm learning them?\n" + "\n".join(trend))
    return "\n".join(out)
