"""Joint FM (joint_contour.update_joint_mask, unchanged) with case-adaptive settings.

* adaptive lam: the base FM settings' lam is an absolute number, while the crossing weights it is compared with
  grow with the strand count squared and with how much the two heads overlap. With `lam_alpha`, lam is set per
  call to  lam = alpha * W0 / S  (W0 = weighted crossings of the start labels, S = strands in the FM nodes, or
  only in the movable nodes with lam_scope="movable"): leaving the drawn label costs each strand alpha x the
  average crossing weight per strand.
* retention guard (`min_retention`): if a head keeps less than min_retention x the strands the start labels gave
  it, lam is multiplied by `guard_factor` and the FM reruns (at most `guard_steps` times).
* node_filter(incidence) -> (n_nodes,) bool: only these FM nodes may move (joint_contour's movable_only with an
  own node set); the others keep their label.
* edge_rows(membership) -> (n*n,) bool: build the projection edges only in these columns.
* slab_dz: depth-aware crossings (slab_edges.py) instead of overlap="projection".
* fast_membership: coarse-to-fine membership (fast_membership.py; same result).

All of it is done by wrapping joint_contour.core's joint_projection_edges / joint_fm_refine during the call, so
joint_contour itself is unchanged. `cache` (a dict per pair of heads, kept by the caller) holds the memberships
and edges so several refinements of the same heads share them.
"""
import contextlib
import time

import numpy as np

from ._util import root_texels


@contextlib.contextmanager
def patched_core(cache, lam_alpha, record, node_filter=None, edge_rows=None, lam_scope="all", slab_dz=None):
    """Wrap joint_contour.core's joint_projection_edges (cache, column restriction, slab) and joint_fm_refine
    (adaptive lam, node restriction) for the duration of an update_joint_mask call."""
    import highlighting.generate_from_3D_models.joint_contour.core as core
    from highlighting.generate_from_3D_models.joint_contour.overlap import joint_graph_objective
    orig_edges, orig_refine = core.joint_projection_edges, core.joint_fm_refine
    edges_cache = cache.setdefault("edges", {})

    def edges_wrapper(grid, membership, *args, **kwargs):
        if edge_rows is not None and "_edges" in record:   # guard reruns within one refine call
            return record["_edges"]
        if edge_rows is not None:
            import scipy.sparse as sp
            from highlighting.generate_from_3D_models.joint_contour.grid import JointMembership
            keep_rows = np.asarray(edge_rows(membership), bool)
            M = (sp.diags(keep_rows.astype(np.int8)) @ membership.M).tocsr()
            M.eliminate_zeros()
            M.sort_indices()
            M = M.astype(bool)
            record["edge_columns"] = int(keep_rows.sum())
            t0 = time.time()
            e = orig_edges(grid, JointMembership(M, membership.n_a, membership.n_b, membership.n), *args, **kwargs)
            record["edge_seconds"] = round(time.time() - t0, 1)
            record["_edges"] = e
            return e
        if slab_dz:   # depth-aware crossings: only pairs crossing at about the same height
            key = ("slab", float(slab_dz))
            if key not in edges_cache:
                from .slab_edges import slab_projection_edges
                t0 = time.time()
                edges_cache[key] = slab_projection_edges(grid, membership, args[0], args[1], dz=slab_dz)
                record["edge_seconds"] = round(time.time() - t0, 1)
            return edges_cache[key]
        key = bool(kwargs.get("depth_weight", False))
        if key not in edges_cache:
            edges_cache[key] = orig_edges(grid, membership, *args, **kwargs)
        return edges_cache[key]

    def refine_wrapper(target, x0, **kwargs):
        if hasattr(target, "nbr"):   # JointGraph (projection overlap)
            w0 = joint_graph_objective(target, x0)
            s = int(target.incidence.node_strands.sum())
            if lam_scope == "movable" and node_filter is not None:
                s = int(target.incidence.node_strands[np.asarray(node_filter(target.incidence), bool)].sum())
            record.update(w0=int(w0), strands=s)
            if lam_alpha is not None:
                kwargs["lam"] = max(1, int(round(lam_alpha * w0 / max(s, 1) * record.get("lam_mult", 1))))
            elif record.get("lam_mult", 1) != 1:
                kwargs["lam"] = max(1, int(round(kwargs.get("lam", 0) * record["lam_mult"])))
            record["lam"] = int(kwargs.get("lam", 0))
        if node_filter is not None:
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


def memberships(grid, heads, cache, fast=False):
    """per head strand/column membership on `grid`, cached in `cache` -> (list, seconds)"""
    res = grid.n
    mkey = ("membership_fast", res) if fast else ("membership", res)
    t0 = time.time()
    if mkey not in cache:
        if fast:
            from .fast_membership import fast_strand_cell_membership
            cache[mkey] = [fast_strand_cell_membership(grid, h["positions"]) for h in heads]
        else:
            from highlighting.generate_from_3D_models.joint_contour.grid import strand_cell_membership
            cache[mkey] = [strand_cell_membership(grid, h["positions"], device="cuda", show_progress=False)
                           for h in heads]
    return cache[mkey], round(time.time() - t0, 1)


def refine_joint_fm(heads, labels, grid, dataset_path, base_kwargs, kwargs=None, cache=None, lam_alpha=None,
                    min_retention=None, guard_factor=4.0, guard_steps=3, node_filter=None, edge_rows=None,
                    lam_scope="all", fast_membership=False, slab_dz=None, attempt_context=None, verbose=False):
    """Joint FM of heads[0] (base) and heads[1] (donor) starting from `labels` ((res, res), 1 = donor).
    base_kwargs: the FM settings (img2hair_tsai_1.JOINT_FM_KWARGS); kwargs: overrides.
    attempt_context: optional callable returning a context manager entered around every update_joint_mask call
    (e.g. to record graph statistics); what it yields last is returned as info["attempt_context"].
    -> (update_joint_mask result dict, info dict)"""
    from highlighting.generate_from_3D_models.joint_contour import update_joint_mask
    cache = {} if cache is None else cache
    res = labels.shape[0]
    base, donor = heads
    mem, membership_seconds = memberships(grid, heads, cache, fast_membership)
    fm_kwargs = dict(base_kwargs, **(kwargs or {}))
    init = [labels[root_texels(h["root_uv"], res)] == i for i, h in enumerate(heads)]
    n_init = [max(int(k.sum()), 1) for k in init]
    t0 = time.time()
    attempts = []
    record = dict(lam_mult=1.0)
    ctx_value = None
    for _ in range(1 + (guard_steps if min_retention else 0)):
        outer = attempt_context() if attempt_context is not None else contextlib.nullcontext()
        with patched_core(cache, lam_alpha, record, node_filter, edge_rows, lam_scope, slab_dz), outer as ctx_value:
            result = update_joint_mask(
                base["positions"], base["root_uv"], donor["positions"], donor["root_uv"], labels == 1,
                dataset_path=dataset_path, device="cuda", grid=grid, method_kwargs=dict(fm_kwargs), verbose=verbose,
                membership_a=mem[0], membership_b=mem[1],
                rowcol_a=root_texels(base["root_uv"], res), rowcol_b=root_texels(donor["root_uv"], res),
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
        fm_seconds=round(time.time() - t0, 2), attempt_context=ctx_value,
        movable_nodes=record.get("movable_nodes"), nodes=record.get("nodes"),
        edge_columns=record.get("edge_columns"), edge_seconds=record.get("edge_seconds"),
        membership_seconds=membership_seconds, fast_membership=fast_membership, slab_dz=slab_dz,
    )
    return result, info
