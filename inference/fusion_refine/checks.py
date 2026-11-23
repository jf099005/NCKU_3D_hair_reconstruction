"""Equality checks of the fast / restricted variants against joint_contour's originals.

    python -m fusion_refine.checks membership <full_strands.npz> [n_strands]
    python -m fusion_refine.checks slab <full_strands_a.npz> <full_strands_b.npz> [n_strands]
(run from inference/, or with inference/ on PYTHONPATH)
"""
import sys
import time

import numpy as np

from . import _ROOT  # noqa: F401  (puts P76154862/ on sys.path)


def _load(npz, n, seed):
    d = np.load(npz)
    P = d["positions"]
    if n is not None and n < len(P):
        P = P[np.sort(np.random.default_rng(seed).choice(len(P), n, replace=False))]
    return np.ascontiguousarray(P, dtype=np.float32)


def _grid():
    import os
    from highlighting.generate_from_3D_models.joint_contour.grid import ScalpGrid
    ds = os.path.join(_ROOT, "DiffLocks_Dataset", "difflocks_dataset")
    return ScalpGrid.from_dataset(256, ds)


def check_membership(npz, n=None):
    from highlighting.generate_from_3D_models.joint_contour.grid import strand_cell_membership
    from .fast_membership import fast_strand_cell_membership
    P, grid = _load(npz, n, 0), _grid()
    t0 = time.time()
    A = strand_cell_membership(grid, P, device="cuda", show_progress=False).astype(bool).tocsr()
    A.sort_indices()
    t1 = time.time()
    B = fast_strand_cell_membership(grid, P)
    t2 = time.time()
    print(f"strands {len(P)}: original {t1 - t0:.1f}s nnz {A.nnz}; fast {t2 - t1:.1f}s nnz {B.nnz}; "
          f"differing entries {(A != B).nnz}")


def check_slab(npz_a, npz_b, n=20000):
    from highlighting.generate_from_3D_models.joint_contour.grid import JointMembership
    from highlighting.generate_from_3D_models.joint_contour.overlap import joint_projection_edges
    from .fast_membership import fast_strand_cell_membership
    from .slab_edges import slab_projection_edges
    pa, pb, grid = _load(npz_a, n, 0), _load(npz_b, n, 1), _grid()
    mem = JointMembership.from_heads(fast_strand_cell_membership(grid, pa), fast_strand_cell_membership(grid, pb), 256)
    ref = joint_projection_edges(grid, mem, pa, pb, show_progress=False)
    big = slab_projection_edges(grid, mem, pa, pb, dz=1.0)
    same = all(np.array_equal(getattr(ref, f), getattr(big, f)) for f in ("ptr", "nbr", "weight"))
    print(f"edges {ref.n_edges}; slab with one slab identical: {same}")
    for dz in (0.02, 0.01, 0.005):
        g = slab_projection_edges(grid, mem, pa, pb, dz=dz)
        print(f"dz {dz * 100:.1f} cm: edges {g.n_edges} ({g.n_edges / ref.n_edges:.2%}), "
              f"weight {g.weight.sum() / ref.weight.sum():.2%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    what, args = sys.argv[1], sys.argv[2:]
    if what == "membership":
        check_membership(args[0], int(args[1]) if len(args) > 1 else None)
    elif what == "slab":
        check_slab(args[0], args[1], int(args[2]) if len(args) > 2 else 20000)
    else:
        raise SystemExit(__doc__)
