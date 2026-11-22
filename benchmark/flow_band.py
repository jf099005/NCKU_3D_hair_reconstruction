"""Direction A: joint FM restricted to a hair-flow-aligned band around the drawn boundary (see LOG.md).

Why: the baseline FM may flip any cell, so on heavily overlapping heads it replaces whole regions (one head
vanishes, or the donor grows into another hairstyle). Crossings come from strands that leave their region along
the hair flow, so the cells worth flipping lie near the drawn boundary and mostly along the flow from it.

1. flow field (texture layout): every strand's growth direction near the root (root -> point at FLOW_ARC m along
   it, both mapped to texels via the nearest root of either head), accumulated as a sign-free structure tensor
   on a coarse grid, smoothed; principal direction f and coherence c in [0, 1].
2. band: Dijkstra from the boundary cells of the drawn mask on the 8-connected texel grid of the hair region.
   A step d (unit vector u, length |d|) costs |d| sqrt(k (u.f)^2 + (u.f_perp)^2), k = 1 - c (1 - 1/a^2):
   with a > 1 moving along a coherent flow costs only 1/a per texel. band = cells with cost <= budget (texels).
   a = 1 is an isotropic band of `budget` texels.
3. joint FM (adaptive_fm.run_variant) with only the nodes whose root cell lies in the band movable; every other
   cell keeps the drawn label.
"""
import numpy as np

import common
from adaptive_fm import run_variant
from fusion_methods import register_method

FLOW_ARC = 0.015       # m along the strand for the growth direction
FLOW_GRID = 64         # structure-tensor grid (texture res / 4)
FLOW_SIGMA = 1.5       # smoothing, in FLOW_GRID cells


def flow_field(heads, res=common.TEXTURE_RES, step=2):
    """(f (res, res, 2) unit direction in (row, col), coherence (res, res))"""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    roots = np.concatenate([h["positions"][:, 0] for h in heads]).astype(np.float64)
    rc = np.concatenate([np.stack(common.root_texels(h["root_uv"], res), 1) for h in heads]).astype(np.float64)
    kd = cKDTree(roots)
    g = FLOW_GRID
    T = np.zeros((g, g, 3))
    for h in heads:
        P = h["positions"][::step].astype(np.float64)
        seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
        s = np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, 1)], 1)
        j = np.clip((s < FLOW_ARC).sum(1), 1, P.shape[1] - 1)
        tip = P[np.arange(len(P)), j]
        _, q = kd.query(tip)
        r0 = np.stack(common.root_texels(h["root_uv"][::step], res), 1).astype(np.float64)
        d = rc[q] - r0
        n = np.linalg.norm(d, axis=1)
        ok = n > 0.5
        d = d[ok] / n[ok, None]
        cell = np.clip((r0[ok] * g / res).astype(int), 0, g - 1)
        for k, v in enumerate((d[:, 0] ** 2, d[:, 0] * d[:, 1], d[:, 1] ** 2)):
            np.add.at(T[..., k], (cell[:, 0], cell[:, 1]), v)
    T = np.stack([gaussian_filter(T[..., k], FLOW_SIGMA) for k in range(3)], -1)
    a, b, c = T[..., 0], T[..., 1], T[..., 2]
    tr = a + c
    disc = np.sqrt(np.maximum((a - c) ** 2 / 4 + b ** 2, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    coh = np.where(tr > 1e-9, (l1 - l2) / np.maximum(l1 + l2, 1e-9), 0)
    ang = 0.5 * np.arctan2(2 * b, a - c)
    f = np.stack([np.cos(ang), np.sin(ang)], -1)          # (row, col) components
    up = lambda x: np.repeat(np.repeat(x, res // g, 0), res // g, 1)  # noqa: E731
    return np.stack([up(f[..., 0]), up(f[..., 1])], -1), up(coh)


def flow_band(labels, region, f, coh, budget, aniso, cap=None):
    """(res, res) bool: hair-region texels within `budget` (texels) of the drawn boundary, distances measured
    with the flow-aligned metric (module docstring)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    res = labels.shape[0]
    idx = -np.ones((res, res), np.int64)
    rr, cc = np.nonzero(region)
    idx[rr, cc] = np.arange(rr.size)
    k = 1 - coh * (1 - 1 / aniso ** 2)
    src, dst, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < res) & (c2 >= 0) & (c2 < res)
        ok[ok] &= region[r2[ok], c2[ok]]
        a, b = idx[rr[ok], cc[ok]], idx[r2[ok], c2[ok]]
        u = np.array([dr, dc], float) / np.hypot(dr, dc)
        fa = 0.5 * (f[rr[ok], cc[ok]] + f[r2[ok], c2[ok]])
        fa /= np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-9)
        along = (fa @ u) ** 2
        kk = 0.5 * (k[rr[ok], cc[ok]] + k[r2[ok], c2[ok]])
        cost = np.hypot(dr, dc) * np.sqrt(kk * along + (1 - along))
        src += [a, b]
        dst += [b, a]
        w += [cost, cost]
    G = coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(rr.size, rr.size)).tocsr()
    lab = labels > 0
    bnd = np.zeros_like(lab)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        sh = np.roll(np.roll(lab, dr, 0), dc, 1)
        shr = np.roll(np.roll(region, dr, 0), dc, 1)
        bnd |= region & shr & (sh != lab)
    srcs = idx[bnd]
    if srcs.size == 0:
        return np.zeros_like(region)
    dist = dijkstra(G, directed=True, indices=srcs, min_only=True, limit=budget + 1e-6)
    band = np.zeros_like(region)
    band[rr, cc] = np.isfinite(dist) & (dist <= budget)
    if cap is not None:   # the band may cover at most `cap` of each head's region: shrink that side's budget
        side = lab[rr, cc]
        for v in (False, True):
            on = side == v
            n_side = int(on.sum())
            d = np.where(np.isfinite(dist), dist, np.inf)[on]
            if n_side and (d <= budget).sum() > cap * n_side:
                lim = np.sort(d)[max(int(cap * n_side) - 1, 0)]
                sel = np.flatnonzero(on)
                band[rr[sel], cc[sel]] = d <= lim
    return band


def flow_smooth(mask, region, f, coh, sel, L=12, iso_sigma=1.0):
    """LIC-style smoothing along the hair flow: every texel of `sel` takes the Gaussian-weighted (sigma L/2)
    average of `mask` along its streamline (+-L unit steps following f, sign kept consistent, only hair-region
    samples count), blended with its own value by the coherence, then a small isotropic blur and threshold 0.5.
    Texels outside `sel` keep `mask`. -> (res, res) bool"""
    from scipy.ndimage import gaussian_filter
    res = mask.shape[0]
    m = mask.astype(np.float64)
    rr, cc = np.nonzero(region)
    acc = m[rr, cc].copy()
    wsum = np.ones(rr.size)
    for sgn in (1.0, -1.0):
        p = np.stack([rr, cc], 1).astype(np.float64)
        d = f[rr, cc] * sgn
        alive = np.ones(rr.size, bool)
        for t in range(1, L + 1):
            q = np.clip(np.rint(p).astype(int), 0, res - 1)
            fd = f[q[:, 0], q[:, 1]]
            fd = np.where(((fd * d).sum(1) < 0)[:, None], -fd, fd)
            d = fd
            p = p + d
            q = np.clip(np.rint(p).astype(int), 0, res - 1)
            alive &= region[q[:, 0], q[:, 1]]
            w = np.exp(-0.5 * (t / (0.5 * L)) ** 2) * alive
            acc += w * m[q[:, 0], q[:, 1]]
            wsum += w
    lic = np.zeros_like(m)
    lic[rr, cc] = acc / wsum
    c = coh
    out = c * lic + (1 - c) * m
    num = gaussian_filter(np.where(region, out, 0.0), iso_sigma)
    den = gaussian_filter(region.astype(np.float64), iso_sigma)
    sm = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0) > 0.5
    return np.where(sel & region, sm, mask.astype(bool))


def keep_from_mask(heads, mask):
    """per head: strands whose root texel the (res, res) bool mask (True = head 1) gives to that head"""
    res = mask.shape[0]
    return [mask[common.root_texels(h["root_uv"], res)] == bool(i) for i, h in enumerate(heads)]


def run_band(heads, labels, ctx, budget, aniso, lam_alpha=None, min_retention=None, smooth_L=None, fm=True,
             cap=None):
    res = labels.shape[0]
    region = common.hair_region([h["root_uv"] for h in heads], res)
    cache_key = ("_afb_flow", id(heads))
    if ctx.get("_afb_key") != cache_key:
        ctx["_afb_key"], ctx["_afb_flow"] = cache_key, flow_field(heads, res)
    f, coh = ctx["_afb_flow"]
    band = flow_band(labels, region, f, coh, budget, aniso, cap)
    band_flat = band.reshape(-1)

    def node_filter(inc):
        return band_flat[inc.node_cell]

    info = dict(band_budget=budget, band_aniso=aniso, band_frac=float(band[region].mean()),
                flow_coherence_mean=float(coh[region].mean()), smooth_L=smooth_L, fm=fm, band_cap=cap)
    if fm:
        fr = run_variant(heads, labels, ctx, lam_alpha=lam_alpha, min_retention=min_retention, guard_factor=2.0,
                         node_filter=node_filter, extra_info=info)
        mask = fr.refined_labels > 0
    else:
        from fusion_methods import FusionResult
        fr, mask = FusionResult(keep=None, info=dict(info)), labels > 0
    if smooth_L:
        mask = flow_smooth(mask, region, f, coh, band, L=smooth_L)
        fr.info["smooth_changed_texels"] = int((mask != (fr.refined_labels > 0 if fm else labels > 0))[region].sum())
    if smooth_L or not fm:
        fr.keep = keep_from_mask(heads, mask)
        fr.refined_labels = mask.astype(np.int16)
    return fr


VARIANTS = {
    # name: (budget texels, anisotropy a, lam_alpha (None = JOINT_FM_KWARGS' lam), min_retention)
    "afb_iso16": (16, 1.0, None, None),
    "afb_iso32": (32, 1.0, None, None),
    "afb_flow16": (16, 3.0, None, None),
    "afb_flow32": (32, 3.0, None, None),
    "afb_flow16_a0.4": (16, 3.0, 0.4, None),
    "afb_flow32_a0.4": (32, 3.0, 0.4, None),
    "afb_flow32_a0.4_g0.7": (32, 3.0, 0.4, 0.7),
    # + flow smoothing (LIC, L texels) after the band FM; "lic_*": smoothing of the drawn mask only, no FM
    "afb_iso32_lic12": (32, 1.0, None, None, 12),
    "afb_flow16_lic12": (16, 3.0, None, None, 12),
    "afb_flow32_lic12": (32, 3.0, None, None, 12),
    "afb_flow32_a0.4_lic12": (32, 3.0, 0.4, None, 12),
    "afb_flow32_lic24": (32, 3.0, None, None, 24),
    # + cap: the band covers at most this share of each head's region (small regions are not swallowed)
    "afb_flow32_lic24_cap0.5": (32, 3.0, None, None, 24, True, 0.5),
    "afb_flow48_lic24_cap0.5": (48, 3.0, None, None, 24, True, 0.5),
    "afb_flow32_a0.4_lic24_cap0.5": (32, 3.0, 0.4, None, 24, True, 0.5),
    "afb_flow32_lic24_cap0.3": (32, 3.0, None, None, 24, True, 0.3),
    "lic_flow32_L12": (32, 3.0, None, None, 12, False),
    "lic_flow32_L24": (32, 3.0, None, None, 24, False),
}


def _make(name, cfg):
    def method(heads, labels, ctx):
        return run_band(heads, labels, ctx, *cfg)
    method.__name__ = method.__qualname__ = name.replace(".", "_")
    method.__doc__ = f"flow_band variant {name}: budget, aniso, lam_alpha, min_retention[, smooth_L, fm, cap] = {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in VARIANTS.items():
    _make(_n, _c)


# ---------------------------------------------------------------- per-case selection between candidates
def run_select(heads, labels, ctx, candidates, min_ret=0.7):
    """Run every candidate method (they share the cached projection edges), keep those whose kept strands
    are >= min_ret of what the drawn mask gives each head, and return the one with the fewest weighted
    projection crossings of the kept strands (the FM's own objective; no evaluation metric is used).
    None qualifies -> the highest retention."""
    from fusion_methods import METHODS
    from highlighting.generate_from_3D_models.joint_contour.overlap import joint_projection_objectives
    init = [labels[common.root_texels(h["root_uv"], labels.shape[0])] == i for i, h in enumerate(heads)]
    n_a = heads[0]["root_uv"].shape[0]
    scored = []
    for name in candidates:
        fr = METHODS[name](heads, labels, ctx)
        edges = ctx["_afm_cache"]["edges"][False]
        state = np.concatenate([~fr.keep[0], fr.keep[1]])
        pairs, weighted = joint_projection_objectives(edges, state, n_a)
        ret = min(float((k & i).sum() / max(int(i.sum()), 1)) for k, i in zip(fr.keep, init))
        scored.append(dict(name=name, fr=fr, crossings=int(weighted), ret=ret))
    ok = [s for s in scored if s["ret"] >= min_ret]
    best = min(ok, key=lambda s: s["crossings"]) if ok else max(scored, key=lambda s: s["ret"])
    fr = best["fr"]
    fr.info = dict(fr.info, selected=best["name"],
                   candidates={s["name"]: dict(crossings=s["crossings"], ret=round(s["ret"], 4)) for s in scored})
    return fr


SELECT_VARIANTS = {
    "afs_guard_or_band": (["afm_a2_guard0.7", "afb_flow32_a0.4_lic24_cap0.5"], 0.7),
    "afs_guard_or_band3": (["afm_a2_guard0.7", "afb_flow32_a0.4_lic24_cap0.5", "afb_flow16_lic12"], 0.7),
}


def _make_select(name, cfg):
    def method(heads, labels, ctx):
        return run_select(heads, labels, ctx, *cfg)
    method.__name__ = method.__qualname__ = name
    method.__doc__ = f"per-case selection {name}: candidates, min_ret = {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in SELECT_VARIANTS.items():
    _make_select(_n, _c)
