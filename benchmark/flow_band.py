"""Benchmark methods of direction A (afb_*, afs_*); the algorithm is in inference/fusion_refine/flow.py.

Direction A: joint FM restricted to a hair-flow-aligned band around the drawn boundary (see LOG.md).

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

from fusion_refine.flow import FLOW_ARC, FLOW_GRID, FLOW_SIGMA, flow_band, flow_field, flow_smooth  # noqa: E402,F401
from fusion_refine._util import keep_from_mask  # noqa: E402


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
