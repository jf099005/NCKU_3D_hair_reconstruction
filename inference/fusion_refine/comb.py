"""Lateral combing of intruding strands after the joint FM (see LOG.md).

Why: the fusion difficulty is driven by intrusion (kept strands lying over the other head's region; Spearman 0.78
with the refined grid score over 28 cases), which is a sideways (scalp-tangent) problem. The radial lift of
strand_edit.py did not help (LOG [7], [8]).

1. joint FM (a FusionResult with keep + refined labels).
2. footprint of every kept strand point = the refined-mask label at the texel of the nearest root (either head).
   A point of head h whose footprint belongs to the other head's region is an intruder; it MIXES only if the other
   head's kept hair is within VOX (3 x 3 x 3 voxels of VOX m) of it -- a point lying clearly above / below the
   other head's hair is a separate layer and is left alone (user rule: same column, different height = allowed).
3. target: the 3D scalp point of the nearest texel inside its own region (region eroded by ERODE texels), as a
   displacement from the footprint's scalp point, tangent only (the radial component removed), capped at d_max.
4. per strand: displacement samples of mixing points, smoothed along the strand, ramped in from the root
   (smoothstep over 15 % of the length); averaged with the K nearest same-head strands (by root) so neighbours
   move together; applied, then follow-the-leader restores the segment lengths, roots stay fixed, turning-angle
   increase limited to 30 deg (strands_editing.stripe_shape.limit_kinks).
Keep / labels never change here.
"""
import time

import numpy as np

from ._util import hair_region, root_texels

STEP = 4          # point subsampling for footprints / displacement
VOX = 0.003       # m, mixing test voxel
ERODE = 2         # texels: targets land this far inside the own region
K_NBR = 8


def _tex_scalp_points(heads, res):
    """(res, res, 3) scalp point per texel: mean root position of its roots, empty texels from the nearest one"""
    from scipy.ndimage import distance_transform_edt
    acc = np.zeros((res, res, 3))
    cnt = np.zeros((res, res))
    for h in heads:
        r, c = root_texels(h["root_uv"], res)
        np.add.at(acc, (r, c), h["positions"][:, 0].astype(np.float64))
        np.add.at(cnt, (r, c), 1)
    has = cnt > 0
    acc[has] /= cnt[has, None]
    _, (ir, ic) = distance_transform_edt(~has, return_indices=True)
    return acc[ir, ic]


def comb(heads, keep, mask, d_max=0.015, smooth=6.0, mix_min=2, max_kink=30.0):
    """-> edited heads, info. mask: (res, res) refined labels (True = donor region)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter1d
    from scipy.spatial import cKDTree
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss
    t0 = time.time()
    res = mask.shape[0]
    region = hair_region([h["root_uv"] for h in heads], res)
    lab = mask.astype(bool)
    tex3d = _tex_scalp_points(heads, res)
    targets = []
    for h in (0, 1):
        own = binary_erosion(region & (lab == bool(h)), iterations=ERODE)
        if not own.any():
            own = region & (lab == bool(h))
        _, (ir, ic) = distance_transform_edt(~own, return_indices=True)
        targets.append((ir, ic))
    roots = np.concatenate([hd["positions"][:, 0] for hd in heads]).astype(np.float64)
    root_rc = np.concatenate([np.stack(root_texels(hd["root_uv"], res), 1) for hd in heads])
    kd = cKDTree(roots)
    center = roots.mean(0)
    # voxel occupancy of each head's kept hair (mixing test)
    allP = np.concatenate([hd["positions"][k][:, ::STEP].reshape(-1, 3) for hd, k in zip(heads, keep)])
    lo = allP.min(0) - 2 * VOX
    dims = np.ceil((allP.max(0) + 2 * VOX - lo) / VOX).astype(int) + 1
    occ = []
    for hd, k in zip(heads, keep):
        g = np.zeros(dims, np.int32)
        v = np.floor((hd["positions"][k][:, ::STEP].reshape(-1, 3) - lo) / VOX).astype(int)
        np.add.at(g, tuple(v.T), 1)
        from scipy.ndimage import uniform_filter
        occ.append(np.rint(uniform_filter(g.astype(np.float32), 3, mode="constant") * 27).astype(np.int32))
    out = [dict(hd) for hd in heads]
    info = dict(d_max_mm=d_max * 1e3)
    for h in (0, 1):
        idx = np.flatnonzero(keep[h])
        if idx.size == 0:
            continue
        P0 = heads[h]["positions"][idx].astype(np.float64)
        X = P0[:, ::STEP]
        S, J, _ = X.shape
        _, q = kd.query(X.reshape(-1, 3))
        fr, fc = root_rc[q, 0], root_rc[q, 1]
        intr = (lab[fr, fc] != bool(h)) & region[fr, fc]
        v = np.clip(np.floor((X.reshape(-1, 3) - lo) / VOX).astype(int), 0, dims - 1)
        mixing = intr & (occ[1 - h][tuple(v.T)] >= mix_min)
        tr, tc = targets[h][0][fr, fc], targets[h][1][fr, fc]
        D = tex3d[tr, tc] - tex3d[fr, fc]
        rad = X.reshape(-1, 3) - center
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
        D -= (D * rad).sum(1, keepdims=True) * rad
        nD = np.linalg.norm(D, axis=1, keepdims=True)
        D *= np.minimum(1.0, d_max / np.maximum(nD, 1e-12))
        D[~mixing] = 0
        D = D.reshape(S, J, 3)
        # along the strand: smooth, so a mixing point drags its neighbours on the strand too
        D = gaussian_filter1d(D, smooth, axis=1, mode="nearest")
        # neighbours (same head, by root) move together
        nb = cKDTree(P0[:, 0]).query(P0[:, 0], k=min(K_NBR, S))[1]
        D = D[nb].mean(1)
        s, seg = dfm.arclength(P0)
        ramp = dfm.smoothstep(s / (0.15 * np.maximum(s[:, -1:], 1e-6)))
        tt = np.minimum(np.arange(P0.shape[1]) / STEP, J - 1)
        i0 = np.floor(tt).astype(int)
        i1 = np.minimum(i0 + 1, J - 1)
        w = (tt - i0)[None, :, None]
        full = D[:, i0] * (1 - w) + D[:, i1] * w
        moved = np.abs(full).max((1, 2)) > 1e-6
        Q = P0 + ramp[..., None] * full
        mi = np.flatnonzero(moved)
        if mi.size:
            Qi = dfm.follow_the_leader(Q[mi], seg[mi])
            Qi[:, 0] = P0[mi, 0]
            Q[mi] = Qi
            Q, nk = ss.limit_kinks(P0, Q, mi, max_kink)
        else:
            nk = 0
        bad = ~np.isfinite(Q).all((1, 2))
        Q[bad] = P0[bad]
        disp = np.linalg.norm(Q - P0, axis=-1).max(1)
        tag = "base" if h == 0 else "donor"
        info.update({f"{tag}_intruding_pts": float(intr.mean()), f"{tag}_mixing_pts": float(mixing.mean()),
                     f"{tag}_moved_strands": int(moved.sum()), f"{tag}_kink_limited": int(nk),
                     f"{tag}_disp_mean_mm": float(disp[moved].mean() * 1e3) if moved.any() else 0.0,
                     f"{tag}_disp_p99_mm": float(np.percentile(disp, 99) * 1e3)})
        pos = heads[h]["positions"].copy()
        pos[idx] = Q.astype(np.float32)
        out[h]["positions"] = pos
    info["comb_seconds"] = round(time.time() - t0, 1)
    return out, info
