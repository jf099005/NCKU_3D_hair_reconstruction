# fusion_refine

Refinement of the two-head per-strand fusion (`img2hair_tsai_*`): which strands of the base / donor head go into
the fused hair, and optionally small strand edits. Developed and measured in `benchmark/`; design notes, results
and the list of usable versions are in `benchmark/LOG.md`, `SUMMARY.md` and `checkpoints.md` (the single source
for numbers, this README only describes the package).

Inputs: heads as `{"positions" (S, P, 3), "root_uv" (S, 2)}` with `root_uv` in the scalp-texture layout, and a
(256, 256) label map / mask in the same layout (0 / False = base). The joint FM itself is
`highlighting/.../joint_contour` (unchanged); this package wraps it.

```python
from fusion_refine import presets
r = presets.refine(heads, labels, "comb20_slab1", grid, dataset_path, img2hair_tsai_1.JOINT_FM_KWARGS)
r["keep"], r["positions"], r["mask"], r["info"]
```

| module | role |
|---|---|
| `presets.py` | named combinations (`guard`, `guard_slab1`, `comb10_slab1`, `comb20_slab1`) |
| `adaptive.py` | joint FM with adaptive lam, retention guard, node / column restrictions, slab crossings, fast membership |
| `fast_membership.py`, `slab_edges.py` | faster membership (same result), depth-aware crossings |
| `comb.py` | lateral combing of intruding strands (moves strands) |
| `flow.py`, `two_stage.py` | flow field / flow-aligned band / LIC smoothing; two-stage helpers (experimental) |
| `checks.py` | equality checks against joint_contour (`python -m fusion_refine.checks membership|slab ...`) |

Benchmark methods (`benchmark/adaptive_fm.py`, `flow_band.py`, `two_stage.py`, `strand_comb.py`) are thin wrappers
around these modules, so changing them changes what the benchmark measures.
