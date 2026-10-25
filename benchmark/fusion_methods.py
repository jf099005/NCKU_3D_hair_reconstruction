"""Fusion methods evaluated by the benchmark.

A method turns m heads + their label map into the strands each head contributes:

    @register_method("name", supported_m=(2,))      # None = any m
    def method(heads, labels, ctx) -> FusionResult

    heads   list of {"positions" (S, P, 3), "root_uv" (S, 2) in the scalp-texture layout}, head 0 = base
    labels  (res, res) int label map (0 = base, k = head k) in the scalp-texture layout
    ctx     dict shared across all cases of a run (dataset_path, caches such as the ScalpGrid, args)

Every method is scored by the same metrics (metrics.py) on its FusionResult.keep, so methods are
directly comparable. "mask_only" (the drawn mask, unrefined) is the reference every refinement should
beat. To benchmark a new fusion algorithm (or an m > 2 one), add a function here.
"""
import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

import common

METHODS = {}


def register_method(name, supported_m=None):
    def deco(fn):
        fn.supported_m = supported_m
        METHODS[name] = fn
        return fn
    return deco


def method_supports(name, m):
    s = METHODS[name].supported_m
    return s is None or m in s


@dataclass
class FusionResult:
    keep: list                       # per head (S_i,) bool: strand goes into the fused hair
    refined_labels: np.ndarray = None  # optional (res, res) label map after refinement
    info: dict = field(default_factory=dict)   # method-specific numbers (JSON-serializable)


def initial_keep(heads, labels):
    """Per head: the strands whose root texel the label map gives to that head."""
    res = labels.shape[0]
    return [labels[common.root_texels(h["root_uv"], res)] == i for i, h in enumerate(heads)]


@register_method("mask_only")
def mask_only(heads, labels, ctx):
    """No refinement: every strand follows the drawn mask at its root."""
    return FusionResult(keep=initial_keep(heads, labels), refined_labels=labels)


def _tsai():
    """inference/img2hair_tsai_1.py, imported lazily (it pulls in mediapipe, k_diffusion, ...)."""
    import img2hair_tsai_1
    return img2hair_tsai_1


def _tsai2():
    """inference/img2hair_tsai_2.py (tsai_1 + trimming), imported lazily."""
    import img2hair_tsai_2
    return img2hair_tsai_2


@contextlib.contextmanager
def _record_overlap_matrix():
    """Sparsity of the overlap matrix the joint FM builds (overlap="projection"), without changing joint_contour:
    update_joint_mask looks joint_projection_edges / build_joint_projection_graph up in joint_contour.core's
    globals, so they are wrapped there for the duration of the call. The wrappers only read the returned
    graphs (no copy, no extra memory) and pass them on unchanged, so the FM result is the same.

    Yields a dict filled by the call (empty with overlap="cell", which builds no overlap matrix):
        n_a, n_b       base / donor strands = the matrix's shape
        nnz            base x donor strand pairs whose projections cross in at least one column
                       (= the 1 elements of the binary overlap matrix)
        density        nnz / (n_a * n_b)  (sparsity = 1 - density)
        weight_sum     sum of the weights (columns in which a pair crosses)
        node_pairs, n_nodes  the same for the FM nodes (cells with node_unit="cell"): node pairs with W != 0
    """
    import highlighting.generate_from_3D_models.joint_contour.core as core
    out = {}
    orig_edges, orig_graph = core.joint_projection_edges, core.build_joint_projection_graph

    def edges_wrapper(grid, membership, *args, **kwargs):
        edges = orig_edges(grid, membership, *args, **kwargs)
        n_a, n_b = int(membership.n_a), int(membership.n_b)
        nnz = int(edges.n_edges)   # symmetric CSR of A-B pairs only: every pair stored twice
        out.update(n_a=n_a, n_b=n_b, nnz=nnz, density=nnz / max(n_a * n_b, 1),
                   weight_sum=int(edges.weight.sum(dtype=np.int64) // 2))
        return edges

    def graph_wrapper(incidence, edges, n_a):
        graph = orig_graph(incidence, edges, n_a)
        out.update(n_nodes=int(graph.n_nodes), node_pairs=int(graph.nbr.shape[0] // 2))
        return graph

    core.joint_projection_edges, core.build_joint_projection_graph = edges_wrapper, graph_wrapper
    try:
        yield out
    finally:
        core.joint_projection_edges, core.build_joint_projection_graph = orig_edges, orig_graph


def _joint_fm_kwargs(tsai, ctx):
    """The script's JOINT_FM_KWARGS (the same object, so nothing changes without run --boundary_fm); with
    ctx["boundary_fm"] a copy with boundary=True (Boundary FM, joint_contour.boundary_fm)."""
    if ctx.get("boundary_fm"):
        return dict(tsai.JOINT_FM_KWARGS, boundary=True)
    return tsai.JOINT_FM_KWARGS


def _fm_info(tsai, result, fm_seconds, overlap_matrix=None, fm_kwargs=None):
    return dict(
        joint_fm_kwargs=tsai.JOINT_FM_KWARGS if fm_kwargs is None else fm_kwargs,
        objectives={k: list(v) if isinstance(v, tuple) else v for k, v in result["objectives"].items()},
        fm_seconds=fm_seconds,
        overlap_matrix=overlap_matrix or None,
    )


def _shared_joint_fm(heads, labels, ctx):
    """The joint FM of img2hair_tsai_2.joint_fm_with_trim before trimming (= tsai_1's update_joint_mask call,
    with the memberships computed outside so the trimming can reuse them), run once per case and cached
    in ctx for tsai_joint_fm and tsai2_joint_fm_trim. Used only when the run includes tsai2_joint_fm_trim."""
    c = ctx.get("_joint_fm_cache")
    if c is not None and c["heads"] is heads and np.array_equal(c["labels"], labels):
        return c
    ctx.pop("_joint_fm_cache", None)
    tsai2 = _tsai2()
    res = labels.shape[0]
    key = ("scalp_grid", res)
    if key not in ctx:
        ctx[key] = tsai2.ScalpGrid.from_dataset(res, ctx["dataset_path"])
    grid, verbose = ctx[key], ctx.get("verbose", False)
    base, donor = heads
    t0 = time.time()
    m_a = tsai2.strand_cell_membership(grid, base["positions"], device="cuda", show_progress=verbose)
    m_b = tsai2.strand_cell_membership(grid, donor["positions"], device="cuda", show_progress=verbose)
    with _record_overlap_matrix() as overlap_matrix:
        result = tsai2.update_joint_mask(
            base["positions"], base["root_uv"], donor["positions"], donor["root_uv"], labels == 1,
            dataset_path=ctx["dataset_path"], device="cuda", grid=grid,
            method_kwargs=_joint_fm_kwargs(tsai2, ctx), verbose=verbose, membership_a=m_a, membership_b=m_b,
            rowcol_a=common.root_texels(base["root_uv"], res), rowcol_b=common.root_texels(donor["root_uv"], res),
        )
    ctx["_joint_fm_cache"] = c = dict(heads=heads, labels=labels, grid=grid, membership_a=m_a, membership_b=m_b,
                                      result=result, fm_seconds=round(time.time() - t0, 2),
                                      overlap_matrix=overlap_matrix)
    return c


@register_method("tsai_joint_fm", supported_m=(2,))
def tsai_joint_fm(heads, labels, ctx):
    """img2hair_tsai_1.refine_mask_per_strand without the image -> hair reconstruction: joint_contour's
    update_joint_mask with the script's own JOINT_FM_KWARGS, each strand looked up at its root texel
    (scalp-texture layout), on a ScalpGrid of the mask resolution."""
    if ctx.get("share_joint_fm"):   # run includes tsai2_joint_fm_trim: one FM per case for both methods
        c = _shared_joint_fm(heads, labels, ctx)
        r = c["result"]
        return FusionResult(keep=[r["visible_a"], r["visible_b"]], refined_labels=r["binary_mask"].astype(np.int16),
                            info=_fm_info(_tsai2(), r, c["fm_seconds"], c["overlap_matrix"],
                                          _joint_fm_kwargs(_tsai2(), ctx)))
    tsai = _tsai()
    res = labels.shape[0]
    key = ("scalp_grid", res)
    if key not in ctx:
        ctx[key] = tsai.ScalpGrid.from_dataset(res, ctx["dataset_path"])
    base, donor = heads
    fm_kwargs = _joint_fm_kwargs(tsai, ctx)
    t0 = time.time()
    with _record_overlap_matrix() as overlap_matrix:
        result = tsai.update_joint_mask(
            base["positions"], base["root_uv"], donor["positions"], donor["root_uv"], labels == 1,
            dataset_path=ctx["dataset_path"], device="cuda", grid=ctx[key],
            method_kwargs=fm_kwargs, verbose=ctx.get("verbose", False),
            rowcol_a=common.root_texels(base["root_uv"], res), rowcol_b=common.root_texels(donor["root_uv"], res),
        )
    info = dict(
        joint_fm_kwargs=fm_kwargs,
        objectives={k: list(v) if isinstance(v, tuple) else v for k, v in result["objectives"].items()},
        fm_seconds=round(time.time() - t0, 2),
        overlap_matrix=overlap_matrix or None,
    )
    return FusionResult(keep=[result["visible_a"], result["visible_b"]],
                        refined_labels=result["binary_mask"].astype(np.int16), info=info)


@register_method("tsai2_joint_fm_trim", supported_m=(2,))
def tsai2_joint_fm_trim(heads, labels, ctx):
    """inference/img2hair_tsai_2.py: tsai_joint_fm + trimming (trim_joint_result) of ctx["trim_percent"] %
    (default img2hair_tsai_2.TRIM_PERCENT) of all strands, the visible ones crossing the other head most."""
    tsai2 = _tsai2()
    c = _shared_joint_fm(heads, labels, ctx)
    pct = ctx.get("trim_percent", tsai2.TRIM_PERCENT)
    t0 = time.time()
    r = tsai2.trim_joint_result(dict(c["result"]), c["grid"], c["membership_a"], c["membership_b"],
                                heads[0]["positions"], heads[1]["positions"], trim_percent=pct,
                                verbose=ctx.get("verbose", False))
    info = _fm_info(tsai2, r, c["fm_seconds"], c["overlap_matrix"], _joint_fm_kwargs(tsai2, ctx))
    info.update(trim=r["trim_info"], trim_seconds=round(time.time() - t0, 2))
    return FusionResult(keep=[r["visible_a"], r["visible_b"]], refined_labels=r["binary_mask"].astype(np.int16),
                        info=info)
