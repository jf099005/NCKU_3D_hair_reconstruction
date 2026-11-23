"""Coarse-to-fine strand/column membership: the same (n*n, S) bool CSR as
joint_contour.grid.strand_cell_membership (GPU path), computed faster (see LOG.md).

The original tests every strand point against every column (chunks of 16 cells x all S*P points). Here:
1. the n x n cells are grouped into B x B blocks; every point is tested once per block against the block's mean
   frame with a tolerance that contains every child column: |t1 / t2 offset| <= half block size + |h| sin(theta) +
   margin (theta = largest angle between the block's mean normal and a child normal; the child columns tilt by up
   to that much over their height), h in [h_min - margin, h_max + margin]. -> candidate points per block.
2. the block's child cells run the original exact test on the candidate points only.
Step 1 only removes points that are in no child column, so the result equals the original (checked with
`python -m fusion_refine.checks membership <npz>`).
"""
import time

import numpy as np
from scipy import sparse

H_MIN, H_MAX = -0.01, 0.4   # joint_contour.grid / update_contour.grid column range


def _block_filter(pts, idx, grid, uu, vv, h_min, h_max, margin, T):
    """indices (subset of idx) of the points that can lie in any column of cells (uu, vv) (module docstring)"""
    import torch
    half = float(grid.half_size)
    o = grid.grid_pos[uu, vv].astype(np.float64)
    nr = grid.grid_normal[uu, vv].astype(np.float64)
    a1 = grid.grid_t1[uu, vv].astype(np.float64)
    a2 = grid.grid_t2[uu, vv].astype(np.float64)
    oc = o.mean(0)
    nc = nr.mean(0)
    nc /= np.linalg.norm(nc)
    t1c = a1.mean(0) - (a1.mean(0) @ nc) * nc
    t1c /= np.linalg.norm(t1c)
    t2c = np.cross(nc, t1c)
    sin_t = float(np.sqrt(np.clip(1 - (nr @ nc).min() ** 2, 0, 1)))
    corner1 = half * (np.abs(a1 @ t1c) + np.abs(a2 @ t1c))
    corner2 = half * (np.abs(a1 @ t2c) + np.abs(a2 @ t2c))
    ext1 = float(np.max(np.abs((o - oc) @ t1c) + corner1))
    ext2 = float(np.max(np.abs((o - oc) @ t2c) + corner2))
    dh = float(np.max(np.abs((o - oc) @ nc))) + max(ext1, ext2) * sin_t
    q = pts[idx] if idx is not None else pts
    rel = q - T(oc)
    h = rel @ T(nc)
    tol = abs(h_max) * sin_t + margin
    cand = (h >= h_min - dh - margin) & (h <= h_max + dh + margin)
    cand &= (rel @ T(t1c)).abs() <= ext1 + h.abs() * sin_t + tol
    cand &= (rel @ T(t2c)).abs() <= ext2 + h.abs() * sin_t + tol
    k = torch.nonzero(cand, as_tuple=True)[0]
    return k if idx is None else idx[k]


def fast_strand_cell_membership(grid, strand_positions, levels=(16, 4), h_min=H_MIN, h_max=H_MAX, margin=1e-3):
    """levels: block sizes, coarse to fine (each divides the previous); the cells of the finest blocks get the exact test."""
    import torch
    P_all = np.asarray(strand_positions)
    S, P, _ = P_all.shape
    n = grid.n
    dtype = P_all.dtype
    dev = torch.device("cuda")
    valid = grid.grid_valid.reshape(n, n)
    half = float(grid.half_size)
    rows, cols = [], []
    T = lambda x: torch.as_tensor(np.asarray(x, dtype), device=dev)  # noqa: E731

    def exact(cand, uu, vv):
        cp = pts[cand]
        o, nr = grid.grid_pos[uu, vv], grid.grid_normal[uu, vv]
        a1, a2 = grid.grid_t1[uu, vv], grid.grid_t2[uu, vv]
        O, N, A1, A2 = T(o), T(nr), T(a1), T(a2)
        strip = ((cp @ A1.T) - (O * A1).sum(1)).abs() <= half
        k_idx, c_loc = torch.nonzero(strip, as_tuple=True)
        r = cp[k_idx] - O[c_loc]
        ok = ((r * A2[c_loc]).sum(1).abs() <= half)
        hh = (r * N[c_loc]).sum(1)
        ok &= (hh >= h_min) & (hh <= h_max)
        C = len(uu)
        key = torch.unique((cand[k_idx[ok]] // P) * C + c_loc[ok])
        rows.append((uu * n + vv)[(key % C).cpu().numpy()])
        cols.append((key // C).cpu().numpy())

    def recurse(cand, u0, v0, size, lv):
        sub = valid[u0:u0 + size, v0:v0 + size]
        if not sub.any() or (cand is not None and cand.numel() == 0):
            return
        uu, vv = np.nonzero(sub)
        uu, vv = uu + u0, vv + v0
        cand = _block_filter(pts, cand, grid, uu, vv, h_min, h_max, margin, T)
        if cand.numel() == 0:
            return
        if lv + 1 < len(levels):
            step = levels[lv + 1]
            for du in range(0, size, step):
                for dv in range(0, size, step):
                    recurse(cand, u0 + du, v0 + dv, step, lv + 1)
        else:
            exact(cand, uu, vv)

    with torch.inference_mode():
        pts = torch.from_numpy(np.ascontiguousarray(P_all.reshape(S * P, 3))).to(dev)
        for bu in range(0, n, levels[0]):
            for bv in range(0, n, levels[0]):
                recurse(None, bu, bv, levels[0], 0)
    r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    M = sparse.csr_matrix((np.ones(r.size, bool), (r, c)), shape=(n * n, S))
    M.sum_duplicates()
    M.sort_indices()
    return M
