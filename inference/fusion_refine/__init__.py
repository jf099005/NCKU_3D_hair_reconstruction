"""Refinement of the two-head per-strand fusion (img2hair_tsai_* family), developed in benchmark/ (see
benchmark/LOG.md, checkpoints.md). Heads are dicts {"positions" (S, P, 3), "root_uv" (S, 2)} with root_uv in
the scalp-texture layout ((row, col) / 256); labels / masks are (256, 256) in that layout (0 / False = base).

Modules
    fast_membership  coarse-to-fine strand/column membership (same result as joint_contour's, ~5x faster)
    slab_edges       depth-aware projection crossings (pairs at clearly different heights do not count)
    adaptive         joint FM with adaptive lam, retention guard and optional node / column restrictions
    comb             lateral combing of intruding strands (moves strands)
    flow             hair-flow field on the scalp texture, flow-aligned boundary band, LIC smoothing
    two_stage        helpers of the coarse (strand subset) -> fine (band) FM
    presets          refine(heads, labels, preset): the named combinations (e.g. "comb20_slab1")

Needs P76154862/ (the folder with highlighting/) on sys.path; importing this package adds it.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
