"""Experimental fusion methods that MOVE strands instead of hiding them (see LOG.md / checkpoints.md).

Idea (from highlighting/strands_editing's rlift_bf): crossings look messy because strands of the two heads
interleave in depth. Instead of dropping the strands that cross (FM), lift one head's strands radially until they
lie on top of the other head's hair (Beer-Lambert optical depth of the other head only, stripe_shape.lift), then
let the other head's strands they passed over move the opposite way (stripe_shape.backfill), so the local density
stays about the same. Every strand of the drawn mask (or of a chosen FM variant) is kept; roots stay fixed.

The "top" head's kept strands are grouped into patches by root texel (PATCH x PATCH texels of the 256 scalp
texture); each patch is lifted as one sheet (lift's stripe = patch, position along it = arc length).

Methods register on import (fusion_methods imports this module); they run only when named in --methods.
"""
import time

import numpy as np

import common
from fusion_methods import METHODS, FusionResult, initial_keep, register_method

PATCH = 16


def _modules():
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss
    from highlighting.generate_from_3D_models.strands_editing import visibility as vz
    return ss, vz, dfm


def _set_density(heads, ctx):
    """visibility model density = the optical render's hair (strands_editing.pipeline.set_density: every strand at
    the sample's DiffLocks radius -> render_density(1, radius)); with a strand subset (run --max_strands) every
    kept strand stands for 1/frac strands (dataset hairstyles have ~95k), so the layering decisions are those of
    the full hair. Radius = mean of the heads' metadata hair_profile_radius (35 um if unknown)."""
    import json
    import os
    _, vz, _ = _modules()
    radii = []
    for h in heads:
        try:
            meta = json.load(open(os.path.join(os.path.dirname(h["npz_path"]), "metadata.json")))
            radii.append(float(meta["hair_profile_radius"]))
        except (KeyError, OSError, ValueError):
            radii.append(0.5 * vz.HAIR_WIDTH)
    frac = min(1.0, np.mean([h["root_uv"].shape[0] for h in heads]) / 95000.0)
    vz.DENSITY_SCALE = float(vz.render_density(1.0, float(np.mean(radii))) / frac)
    return vz.DENSITY_SCALE


def _patch_ids(root_uv, res=common.TEXTURE_RES, patch=PATCH):
    r, c = common.root_texels(root_uv, res)
    key = (r // patch) * (res // patch + 1) + c // patch
    _, ids = np.unique(key, return_inverse=True)
    return ids + 1


def lift_fuse(heads, keep, top, backfill=True, tau_ok=0.5, d_max=0.025, max_kink=30.0, patch=PATCH):
    """keep: per head kept strands; top: index of the head lifted over the other. -> edited heads, info"""
    ss, vz, dfm = _modules()
    bot = 1 - top
    it, ib = np.flatnonzero(keep[top]), np.flatnonzero(keep[bot])
    P0 = np.concatenate([heads[top]["positions"][it], heads[bot]["positions"][ib]]).astype(np.float64)
    ids = np.zeros(len(P0), np.int64)
    ids[:it.size] = _patch_ids(heads[top]["root_uv"][it], patch=patch)
    length = np.linalg.norm(np.diff(P0, axis=1), axis=-1).sum(1)
    ids[length < 0.01] = 0                     # (near) zero-length strands: nothing to lift, 0/0 in lift's ramp
    center = P0[:, 0].mean(0)
    t0 = time.time()
    tau = ss.base_tau(P0, ids == 0, center)
    Q, lifts = ss.lift(P0, ids, np.zeros(len(P0)), center, tau=tau, tau_ok=tau_ok, d_max=d_max)
    idx = np.flatnonzero(ids > 0)
    _, seg0 = dfm.arclength(P0[idx])
    Qi = dfm.follow_the_leader(Q[idx], seg0)
    Qi[:, 0] = P0[idx, 0]
    Q[idx] = Qi
    Q, nk = ss.limit_kinks(P0, Q, idx, max_kink)
    info = dict(top=int(top), lift_seconds=round(time.time() - t0, 1), kink_limited=int(nk),
                lift_mean_mm=float(np.mean([x.mean() for x in lifts.values()]) * 1e3) if lifts else 0.0,
                lift_max_mm=float(np.max([x.max() for x in lifts.values()]) * 1e3) if lifts else 0.0)
    if backfill:
        t0 = time.time()
        Q, partner, bi = ss.backfill(P0, Q, ids)
        info.update(bi, backfill_seconds=round(time.time() - t0, 1))
    bad = ~np.isfinite(Q).all((1, 2))
    Q[bad] = P0[bad]
    info["nan_reverted"] = int(bad.sum())
    move = np.linalg.norm(Q - P0, axis=-1).max(1)
    info.update(moved_top_mm_p95=float(np.percentile(move[:it.size], 95) * 1e3) if it.size else 0.0,
                moved_bot_mm_p95=float(np.percentile(move[it.size:], 95) * 1e3) if ib.size else 0.0)
    out = [dict(h) for h in heads]
    for h in out:
        h["positions"] = h["positions"].copy()
    out[top]["positions"][it] = Q[:it.size].astype(np.float32)
    out[bot]["positions"][ib] = Q[it.size:].astype(np.float32)
    return out, info


def _top_auto(heads, keep):
    """the head that is already more on top: lower median optical depth of the other head above its points"""
    ss, _, _ = _modules()
    P = [h["positions"][k].astype(np.float64) for h, k in zip(heads, keep)]
    allP = np.concatenate(P)
    center = allP[:, 0].mean(0)
    med = []
    for t in (0, 1):
        o = 1 - t
        both = np.concatenate([P[t], P[o]])
        tau, origin, voxel = ss.base_tau(both, np.r_[np.zeros(len(P[t]), bool), np.ones(len(P[o]), bool)], center)
        X = P[t][:: max(1, len(P[t]) // 2000), ::8].reshape(-1, 3)
        med.append(float(np.median(ss.vz._sample(tau, X[:, None], origin, voxel)[:, 0])))
    return int(np.argmin(med)), med


def point_region(heads, keep, labels, step=4):
    """per head: (S_kept, P//step) region label (0 = base, 1 = donor) of the strand points, from the drawn mask's
    label at the root texel of the nearest root of either head (a footprint on the scalp)"""
    from scipy.spatial import cKDTree
    res = labels.shape[0]
    roots = np.concatenate([h["positions"][:, 0] for h in heads]).astype(np.float64)
    lab = np.concatenate([labels[common.root_texels(h["root_uv"], res)] for h in heads])
    kd = cKDTree(roots)
    out = []
    for h, k in zip(heads, keep):
        X = h["positions"][k][:, ::step].astype(np.float64)
        _, q = kd.query(X.reshape(-1, 3), k=1)
        out.append(lab[q].reshape(X.shape[:2]))
    return out


def region_lift(heads, keep, labels, backfill=True, tau_ok=0.5, d_max=0.025, max_kink=30.0, patch=PATCH, step=4):
    """region-owner on top: for each head h, its strands are lifted over only the OTHER head's points that lie in
    h's region (footprint = point_region), then (backfill) those intruders move the opposite way. Head 0 first,
    then head 1 on the updated geometry. -> edited heads, info"""
    ss, vz, dfm = _modules()
    out = [dict(h) for h in heads]
    for h in out:
        h["positions"] = h["positions"].copy()
    info = {}
    for top in (0, 1):
        bot = 1 - top
        it, ib = np.flatnonzero(keep[top]), np.flatnonzero(keep[bot])
        P0 = np.concatenate([out[top]["positions"][it], out[bot]["positions"][ib]]).astype(np.float64)
        ids = np.zeros(len(P0), np.int64)
        ids[:it.size] = _patch_ids(out[top]["root_uv"][it], patch=patch)
        length = np.linalg.norm(np.diff(P0, axis=1), axis=-1).sum(1)
        ids[length < 0.01] = 0
        reg = point_region(out, keep, labels, step=step)
        # optical depth of the intruders only: bottom-head points in the top head's region (the others are
        # moved far away so they deposit outside the voxel grid's influence: zero-weight copy)
        Pb = P0.copy()
        intr = np.repeat(reg[bot] == top, step, axis=1)[:, :P0.shape[1]]
        bot_pts = Pb[it.size:]
        bot_pts[~intr] = np.nan
        center = P0[:, 0].mean(0)
        tau = _tau_points(ss, vz, P0, bot_pts, center)
        t0 = time.time()
        Q, lifts = ss.lift(P0, ids, np.zeros(len(P0)), center, tau=tau, tau_ok=tau_ok, d_max=d_max)
        idx = np.flatnonzero(ids > 0)
        _, seg0 = dfm.arclength(P0[idx])
        Qi = dfm.follow_the_leader(Q[idx], seg0)
        Qi[:, 0] = P0[idx, 0]
        Q[idx] = Qi
        Q, nk = ss.limit_kinks(P0, Q, idx, max_kink)
        tag = "base" if top == 0 else "donor"
        info[f"{tag}_lift_max_mm"] = float(np.max([x.max() for x in lifts.values()]) * 1e3) if lifts else 0.0
        info[f"{tag}_kink_limited"] = int(nk)
        info[f"{tag}_intruder_pts"] = int(intr.sum())
        if backfill:
            Q, partner, bi = ss.backfill(P0, Q, ids)
            info[f"{tag}_n_paired"] = bi.get("n_paired", 0)
        bad = ~np.isfinite(Q).all((1, 2))
        Q[bad] = P0[bad]
        move = np.linalg.norm(Q - P0, axis=-1).max(1)
        info[f"{tag}_moved_mm_p95"] = float(np.percentile(move[:it.size], 95) * 1e3) if it.size else 0.0
        info[f"{tag}_seconds"] = round(time.time() - t0, 1)
        out[top]["positions"][it] = Q[:it.size].astype(np.float32)
        out[bot]["positions"][ib] = Q[it.size:].astype(np.float32)
    return out, info


def _tau_points(ss, vz, P_all, pts, center, voxel=0.002, pad=0.04, tau_max=30.0):
    """ss.base_tau for an arbitrary point set (strands with NaN points: those segments are skipped)"""
    lo = P_all.reshape(-1, 3).min(0) - pad
    hi = P_all.reshape(-1, 3).max(0) + pad
    shape = np.ceil((hi - lo) / voxel).astype(np.int64)
    origin = lo.astype(np.float64)
    sig = np.zeros(tuple(shape), np.float32)
    dummy = np.zeros(tuple(shape), np.float32)
    # NaN segments: _deposit skips nothing by itself, so split the strands at NaN runs by zeroing them out
    # far outside the grid (deposits outside the grid are dropped by its bounds check)
    far = np.where(np.isfinite(pts), pts, -1e3)
    vz._deposit(np.ascontiguousarray(far, np.float32), origin, float(voxel), shape, np.asarray(center, np.float64),
                vz.FRONT, sig, dummy)
    sig *= vz.DENSITY_SCALE * vz.HAIR_WIDTH * -np.log1p(-vz.HAIR_OPACITY) / voxel ** 3
    tau = vz._march(sig, origin, float(voxel), np.asarray(center, np.float64), vz.FRONT, True, tau_max)
    return tau, origin, float(voxel)


VARIANTS = {
    # name: (keep_from, top, backfill)
    "edit_lift_donor": ("mask_only", 1, True),
    "edit_lift_base": ("mask_only", 0, True),
    "edit_lift_auto": ("mask_only", "auto", True),
    "edit_lift_auto_nobf": ("mask_only", "auto", False),
    "edit_guardfm_lift_auto": ("afm_a2_guard0.7", "auto", True),
    "edit_region_lift": ("mask_only", "region", True),
    "edit_region_lift_nobf": ("mask_only", "region", False),
    "edit_guardfm_region_lift": ("afm_a2_guard0.7", "region", True),
}


def run_variant(heads, labels, ctx, keep_from, top, backfill):
    _set_density(heads, ctx)
    t0 = time.time()
    if keep_from == "mask_only":
        keep, info0 = initial_keep(heads, labels), {}
    else:
        fr = METHODS[keep_from](heads, labels, ctx)
        keep, info0 = fr.keep, dict(keep_from=keep_from, fm_lam=fr.info.get("lam"))
    info = dict(info0)
    if top == "region":
        edited, li = region_lift(heads, keep, labels, backfill=backfill)
        info.update(li, total_seconds=round(time.time() - t0, 1))
        return FusionResult(keep=keep, info=info, heads=edited)
    if top == "auto":
        top, med = _top_auto(heads, keep)
        info["auto_tau_median"] = med
    edited, li = lift_fuse(heads, keep, top, backfill=backfill)
    info.update(li, total_seconds=round(time.time() - t0, 1))
    return FusionResult(keep=keep, info=info, heads=edited)


def _make(name, cfg):
    def method(heads, labels, ctx):
        return run_variant(heads, labels, ctx, *cfg)
    method.__name__ = method.__qualname__ = name
    method.__doc__ = f"strand_edit variant {name}: keep_from, top, backfill = {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in VARIANTS.items():
    _make(_n, _c)
