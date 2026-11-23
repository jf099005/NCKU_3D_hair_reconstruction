"""Benchmark methods of the lateral combing (comb*); the algorithm is in inference/fusion_refine/comb.py.

Lateral combing of intruding strands after the joint FM (see LOG.md).

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
Keep / labels never change here; `refit` reruns the method's FM once on the combed geometry.
"""
import numpy as np

import common  # noqa: F401
from fusion_methods import METHODS, FusionResult, register_method

from fusion_refine.comb import comb  # noqa: E402


def run_comb(heads, labels, ctx, base_method, d_max, refit=False):
    fr = METHODS[base_method](heads, labels, ctx)
    mask = fr.refined_labels > 0
    edited, info = comb(heads, fr.keep, mask, d_max=d_max)
    info = dict(base_method=base_method, base_info={k: fr.info.get(k) for k in ("lam", "attempts", "slab_dz")}, **info)
    keep = fr.keep
    if refit:   # rerun the FM on the combed geometry (fresh per-case cache: positions changed)
        ctx2 = dict(dataset_path=ctx["dataset_path"])
        fr2 = METHODS[base_method](edited, labels, ctx2)
        keep, mask = fr2.keep, fr2.refined_labels > 0
        info["refit_lam"] = fr2.info.get("lam")
    return FusionResult(keep=keep, refined_labels=mask.astype(np.int16), info=info, heads=edited)


VARIANTS = {
    # name: (base FM method, d_max m, refit)
    "comb10_guard": ("afm_a2_guard0.7_fast", 0.010, False),
    "comb20_guard": ("afm_a2_guard0.7_fast", 0.020, False),
    "comb10_guard_refit": ("afm_a2_guard0.7_fast", 0.010, True),
    "comb10_slab1": ("afm_a2_guard0.7_slab1", 0.010, False),
    "comb20_slab1": ("afm_a2_guard0.7_slab1", 0.020, False),
}


def _make(name, cfg):
    def method(heads, labels, ctx):
        return run_comb(heads, labels, ctx, *cfg)
    method.__name__ = method.__qualname__ = name
    method.__doc__ = f"lateral comb {name}: base_method, d_max, refit = {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in VARIANTS.items():
    _make(_n, _c)
