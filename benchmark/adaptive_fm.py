"""Benchmark methods of the joint FM with case-adaptive settings (afm_*). The algorithm lives in
inference/fusion_refine/adaptive.py (adaptive lam, retention guard, node / column restrictions, slab crossings,
fast membership); this module only registers the variants and keeps the per-case cache.

Registered into fusion_methods.METHODS on import (fusion_methods imports this module at the end). They run
only when named in `run --methods`, so the default run is unchanged.
"""
import numpy as np

import common  # noqa: F401  (sys.path: repo, inference/, P76154862/)
from fusion_methods import FusionResult, _record_overlap_matrix, _tsai, register_method
from fusion_refine import adaptive as fr_adaptive

# name -> settings. kwargs: overrides of JOINT_FM_KWARGS; lam_alpha: adaptive lam (None = keep kwargs' lam);
# min_retention / guard_factor / guard_steps: retention guard (None = off).
VARIANTS = {
    "afm_a0.05": dict(lam_alpha=0.05),
    "afm_a0.2": dict(lam_alpha=0.2),
    "afm_a0.4": dict(lam_alpha=0.4),
    "afm_a0.5": dict(lam_alpha=0.5),
    "afm_a1": dict(lam_alpha=1.0),
    "afm_a2": dict(lam_alpha=2.0),
    "afm_a5": dict(lam_alpha=5.0),
    "afm_a1_guard0.7": dict(lam_alpha=1.0, min_retention=0.7, guard_factor=2.0),
    "afm_a2_guard0.7": dict(lam_alpha=2.0, min_retention=0.7, guard_factor=2.0),
    "afm_a2_guard0.7_fast": dict(lam_alpha=2.0, min_retention=0.7, guard_factor=2.0, fast_membership=True),
    # depth-aware crossings: base / donor strands in the same column but > ~dz apart in height do not count
    "afm_a2_guard0.7_slab1": dict(lam_alpha=2.0, min_retention=0.7, guard_factor=2.0, fast_membership=True,
                                  slab_dz=0.01),
    "afm_a2_guard0.7_slab2": dict(lam_alpha=2.0, min_retention=0.7, guard_factor=2.0, fast_membership=True,
                                  slab_dz=0.02),
    "afm_bal0.05": dict(kwargs=dict(balance_tol=0.05)),
    "afm_guard0.5": dict(min_retention=0.5),
    "afm_a0.2_guard0.5": dict(lam_alpha=0.2, min_retention=0.5),
    "afm_depth": dict(kwargs=dict(overlap="projection_depth")),
    "afm_depth_a0.2": dict(kwargs=dict(overlap="projection_depth"), lam_alpha=0.2),
}


def _case_cache(heads, ctx):
    """Per-case cache (memberships, projection edges), reset when the heads change."""
    c = ctx.get("_afm_cache")
    if c is None or c["heads"] is not heads:
        ctx["_afm_cache"] = c = dict(heads=heads, edges={})
    return c


def run_variant(heads, labels, ctx, kwargs=None, lam_alpha=None, min_retention=None, guard_factor=4.0,
                guard_steps=3, node_filter=None, extra_info=None, edge_rows=None, lam_scope="all",
                fast_membership=False, slab_dz=None):
    """fusion_refine.adaptive.refine_joint_fm with img2hair_tsai_1.JOINT_FM_KWARGS as the base settings."""
    tsai = _tsai()
    res = labels.shape[0]
    key = ("scalp_grid", res)
    if key not in ctx:
        ctx[key] = tsai.ScalpGrid.from_dataset(res, ctx["dataset_path"])
    result, info = fr_adaptive.refine_joint_fm(
        heads, labels, ctx[key], ctx["dataset_path"], tsai.JOINT_FM_KWARGS, kwargs=kwargs,
        cache=_case_cache(heads, ctx), lam_alpha=lam_alpha, min_retention=min_retention, guard_factor=guard_factor,
        guard_steps=guard_steps, node_filter=node_filter, edge_rows=edge_rows, lam_scope=lam_scope,
        fast_membership=fast_membership, slab_dz=slab_dz, attempt_context=_record_overlap_matrix,
        verbose=ctx.get("verbose", False))
    info["overlap_matrix"] = info.pop("attempt_context") or None
    info.update(extra_info or {})
    return FusionResult(keep=[result["visible_a"], result["visible_b"]],
                        refined_labels=result["binary_mask"].astype(np.int16), info=info)


def _make(name, cfg):
    def method(heads, labels, ctx):
        return run_variant(heads, labels, ctx, **cfg)
    method.__name__ = method.__qualname__ = name.replace(".", "_")
    method.__doc__ = f"adaptive_fm variant {name}: {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in VARIANTS.items():
    _make(_n, _c)
