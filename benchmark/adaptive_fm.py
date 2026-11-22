"""Experimental fusion methods: joint FM with case-adaptive settings (see LOG.md / checkpoints.md).

Registered into fusion_methods.METHODS on import (fusion_methods imports this module at the end). They run
only when named in `run --methods`, so the default run is unchanged.

All variants call img2hair_tsai_1.update_joint_mask (joint_contour, unchanged) with
img2hair_tsai_1.JOINT_FM_KWARGS overridden per variant. Two things are added on the benchmark side only:

* per-case caches of the strand/cell memberships and of the projection edges (the expensive part), so
  several variants of one case share them. They are dropped when the next case starts.
* adaptive lam: JOINT_FM_KWARGS' lam is an absolute number, while the crossing weights it is compared
  with grow with the strand count squared and with how much the two heads overlap (highlighting
  Summary: "FM 的 lam 是絕對值"). With `lam_alpha`, lam is set per case to
        lam = alpha * W0 / S
  W0 = weighted crossing pairs of the drawn mask (the FM objective at its start), S = strands in the
  FM nodes, i.e. leaving the drawn label costs each strand alpha x the average crossing weight per strand.
  It is set by wrapping joint_contour.core.joint_fm_refine during the call (graph target only).
* retention guard (`min_retention`): if a head keeps less than min_retention x the strands the drawn mask
  gave it, lam is multiplied by `guard_factor` and the FM reruns (at most `guard_steps` times).
"""
import contextlib
import time

import numpy as np

import common
from fusion_methods import FusionResult, _record_overlap_matrix, _tsai, register_method

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


@contextlib.contextmanager
def _patched_core(cache, lam_alpha, record, node_filter=None):
    """Wrap joint_contour.core's joint_projection_edges (cache) and joint_fm_refine (adaptive lam)."""
    import highlighting.generate_from_3D_models.joint_contour.core as core
    from highlighting.generate_from_3D_models.joint_contour.overlap import joint_graph_objective
    orig_edges, orig_refine = core.joint_projection_edges, core.joint_fm_refine

    def edges_wrapper(grid, membership, *args, **kwargs):
        key = bool(kwargs.get("depth_weight", False))
        if key not in cache["edges"]:
            cache["edges"][key] = orig_edges(grid, membership, *args, **kwargs)
        return cache["edges"][key]

    def refine_wrapper(target, x0, **kwargs):
        if hasattr(target, "nbr"):   # JointGraph (projection overlap)
            w0 = joint_graph_objective(target, x0)
            s = int(target.incidence.node_strands.sum())
            record.update(w0=int(w0), strands=s)
            if lam_alpha is not None:
                kwargs["lam"] = max(1, int(round(lam_alpha * w0 / max(s, 1) * record.get("lam_mult", 1))))
            elif record.get("lam_mult", 1) != 1:
                kwargs["lam"] = max(1, int(round(kwargs.get("lam", 0) * record["lam_mult"])))
            record["lam"] = int(kwargs.get("lam", 0))
        if node_filter is not None:   # only these nodes may move (joint_contour's movable_only, own node set)
            inc = target.incidence
            orig_mov = inc.node_movable
            inc.node_movable = np.asarray(node_filter(inc), bool)
            record["movable_nodes"] = int(inc.node_movable.sum())
            record["nodes"] = int(inc.n_nodes)
            try:
                return orig_refine(target, x0, **dict(kwargs, movable_only=True))
            finally:
                inc.node_movable = orig_mov
        return orig_refine(target, x0, **kwargs)

    core.joint_projection_edges, core.joint_fm_refine = edges_wrapper, refine_wrapper
    try:
        yield
    finally:
        core.joint_projection_edges, core.joint_fm_refine = orig_edges, orig_refine


def run_variant(heads, labels, ctx, kwargs=None, lam_alpha=None, min_retention=None, guard_factor=4.0,
                guard_steps=3, node_filter=None, extra_info=None):
    tsai = _tsai()
    res = labels.shape[0]
    key = ("scalp_grid", res)
    if key not in ctx:
        ctx[key] = tsai.ScalpGrid.from_dataset(res, ctx["dataset_path"])
    grid = ctx[key]
    cache = _case_cache(heads, ctx)
    base, donor = heads
    if "membership" not in cache:
        from highlighting.generate_from_3D_models.joint_contour.grid import strand_cell_membership
        cache["membership"] = [strand_cell_membership(grid, h["positions"], device="cuda", show_progress=False)
                               for h in heads]
    fm_kwargs = dict(tsai.JOINT_FM_KWARGS, **(kwargs or {}))
    init = [labels[common.root_texels(h["root_uv"], res)] == i for i, h in enumerate(heads)]
    n_init = [max(int(k.sum()), 1) for k in init]
    t0 = time.time()
    attempts = []
    record = dict(lam_mult=1.0)
    for step in range(1 + (guard_steps if min_retention else 0)):
        with _patched_core(cache, lam_alpha, record, node_filter), _record_overlap_matrix() as overlap_matrix:
            result = tsai.update_joint_mask(
                base["positions"], base["root_uv"], donor["positions"], donor["root_uv"], labels == 1,
                dataset_path=ctx["dataset_path"], device="cuda", grid=grid, method_kwargs=dict(fm_kwargs),
                verbose=ctx.get("verbose", False),
                membership_a=cache["membership"][0], membership_b=cache["membership"][1],
                rowcol_a=common.root_texels(base["root_uv"], res), rowcol_b=common.root_texels(donor["root_uv"], res),
            )
        keep = [result["visible_a"], result["visible_b"]]
        ret = [float((k & i).sum() / n) for k, i, n in zip(keep, init, n_init)]
        attempts.append(dict(lam=record.get("lam"), retention_vs_mask=ret))
        if not min_retention or min(ret) >= min_retention:
            break
        record["lam_mult"] *= guard_factor
    info = dict(
        joint_fm_kwargs=fm_kwargs, lam_alpha=lam_alpha, min_retention=min_retention,
        lam=record.get("lam"), w0=record.get("w0"), fm_strands=record.get("strands"), attempts=attempts,
        objectives={k: list(v) if isinstance(v, tuple) else v for k, v in result["objectives"].items()},
        fm_seconds=round(time.time() - t0, 2), overlap_matrix=overlap_matrix or None,
        movable_nodes=record.get("movable_nodes"), nodes=record.get("nodes"), **(extra_info or {}),
    )
    return FusionResult(keep=keep, refined_labels=result["binary_mask"].astype(np.int16), info=info)


def _make(name, cfg):
    def method(heads, labels, ctx):
        return run_variant(heads, labels, ctx, **cfg)
    method.__name__ = method.__qualname__ = name.replace(".", "_")
    method.__doc__ = f"adaptive_fm variant {name}: {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in VARIANTS.items():
    _make(_n, _c)
