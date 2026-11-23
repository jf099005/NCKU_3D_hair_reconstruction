"""Depth-aware projection crossings ("slab" overlap; see LOG.md).

joint_contour's overlap="projection" counts a base / donor strand pair as crossing in a column when their
projections onto the column's tangent plane intersect, at ANY height. Two strands that share a column but lie at
clearly different heights (one layer above the other) do not interleave visually, so (user request) such pairs
are allowed: here a pair crosses in a column only if their projections intersect among segments of the same
height slab. Slabs are [k*dz/2, k*dz/2 + dz] along the cell normal (half-overlapping, so two segments less than
~dz apart in height always share one); a segment belongs to every slab its height interval overlaps.
Weight = number of columns with such a crossing (as the original).

Uses joint_contour / update_contour's numba primitives unchanged (project_segments, crossing_keys,
weighted_pairs, symmetric_csr). With dz >= the whole column height it returns exactly
joint_projection_edges (checked by `python -m fusion_refine.checks slab <npz_a> <npz_b>`).
"""
import numpy as np

H_MIN, H_MAX = -0.01, 0.4


def slab_projection_edges(grid, membership, positions_a, positions_b, dz=0.01, h_min=H_MIN, h_max=H_MAX,
                          n_bins=None, n_threads=None, show_progress=False):
    from highlighting.generate_from_3D_models.update_contour.overlap import (
        StrandGraph, cell_frame, crossing_keys, project_segments, symmetric_csr, weighted_pairs)
    pos_a = np.ascontiguousarray(positions_a)
    pos_b = np.ascontiguousarray(positions_b)
    M, n_a, n = membership.M, membership.n_a, membership.n
    S = membership.n_strands
    half_size = float(grid.half_size)
    rows = []
    for r in range(M.shape[0]):
        idx = M.indices[M.indptr[r]:M.indptr[r + 1]]
        if idx.size >= 2 and idx[0] < n_a <= idx[-1]:
            rows.append(r)
    step = dz / 2

    def work(r):
        idx = M.indices[M.indptr[r]:M.indptr[r + 1]].astype(np.int64)
        a_ids, b_ids = idx[idx < n_a], idx[idx >= n_a]
        frame = cell_frame(grid.grid_pos, grid.grid_t1, grid.grid_t2, grid.grid_normal, *divmod(r, n))
        seg_a = project_segments(pos_a, a_ids, a_ids, *frame, half_size, float(h_min), float(h_max))
        seg_b = project_segments(pos_b, b_ids - n_a, b_ids, *frame, half_size, float(h_min), float(h_max))
        sx, sy, ex, ey, owner, sz, ez = [np.concatenate([sa, sb]) for sa, sb in zip(seg_a, seg_b)]
        if sx.size == 0:
            return np.empty(0, np.int64)
        lo, hi = np.minimum(sz, ez), np.maximum(sz, ez)
        isb = owner >= n_a
        k0 = int(np.floor((lo.min() - h_min) / step)) - 1
        k1 = int(np.floor((hi.max() - h_min) / step))
        if (k1 - k0) * step <= dz:      # the whole column fits in one slab: same as the original
            return crossing_keys(sx, sy, ex, ey, owner, half_size, int(n_bins or 0), S, n_a)
        out = []
        for k in range(max(k0, -1), k1 + 1):
            s_lo = h_min + k * step
            sel = (lo <= s_lo + dz) & (hi >= s_lo)
            if sel.sum() < 2:
                continue
            b = isb[sel]
            if b.all() or not b.any():
                continue
            out.append(crossing_keys(sx[sel], sy[sel], ex[sel], ey[sel], owner[sel], half_size, int(n_bins or 0),
                                     S, n_a))
        if not out:
            return np.empty(0, np.int64)
        return np.unique(np.concatenate(out))   # one entry per pair per column

    keys, weights = weighted_pairs(work, rows, n_threads, show_progress, desc="slab projection overlap")
    return StrandGraph(*symmetric_csr(keys, weights, S))
