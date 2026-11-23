"""Benchmark methods of the two-stage FM (ts_*); helpers in inference/fusion_refine/two_stage.py.

Two-stage (coarse -> fine) joint FM: avoid the full projection-crossing graph (see LOG.md).

The full graph (every base x donor strand pair whose projections cross) is ~5 min / 12 GB per full-strand case.
1. coarse: afm_a2_guard0.7 on a random subset of N1 strands per head, on an RES1 x RES1 mask / column grid
   (labels = block majority of the drawn mask). ~1 s. Its changed cells, upsampled, give the region to refine.
2. fine: every strand, starting from the stage-1 mask; FM may only move nodes rooted in the band = the stage-1
   changed region plus the stage-1 boundary, dilated by W texels; the projection edges are built only in the
   columns that band strands pass through (crossings between two strands outside the band cannot change any
   move, so the optimum is the same while most of the graph is skipped). Adaptive lam (alpha 2) normalized by the
   band's strands, retention guard 0.7.
"""
import time

import numpy as np

import common
from adaptive_fm import run_variant
from fusion_methods import register_method


from fusion_refine.two_stage import block_majority as _block_majority, boundary as _boundary, upsample as _up  # noqa: E402,F401


def run_two_stage(heads, labels, ctx, n1=5000, res1=64, width=8, alpha=2.0, min_ret=0.7, fast=False):
    from scipy.ndimage import binary_dilation
    res = labels.shape[0]
    t0 = time.time()
    # stage 1: subset, coarse grid
    sub = [common.load_hairstyle(h["npz_path"], n1, seed=7919 + i) for i, h in enumerate(heads)]
    lab_c = _block_majority(labels, res1)
    fr1 = run_variant(sub, lab_c, dict(dataset_path=ctx["dataset_path"]), lam_alpha=alpha, min_retention=min_ret,
                      guard_factor=2.0)
    k = res // res1
    ch_c = (fr1.refined_labels > 0) != (lab_c > 0)
    mask1 = np.where(_up(ch_c, k), _up(fr1.refined_labels > 0, k), labels > 0)
    t1 = time.time() - t0
    # stage 2: all strands, band only
    region = common.hair_region([h["root_uv"] for h in heads], res)
    band = binary_dilation(_up(ch_c, k) | _boundary(mask1, region), iterations=width) & region
    band_flat = band.reshape(-1)
    rc = [common.root_texels(h["root_uv"], res) for h in heads]
    in_band = np.concatenate([band[r, c] for r, c in rc])

    def node_filter(inc):
        return band_flat[inc.node_cell]

    def edge_rows(membership):
        return np.asarray(membership.M @ in_band.astype(np.int64)).ravel() > 0

    fr = run_variant(heads, mask1.astype(labels.dtype), ctx, lam_alpha=alpha, min_retention=min_ret,
                     guard_factor=2.0, node_filter=node_filter, edge_rows=edge_rows, lam_scope="movable",
                     fast_membership=fast,
                     extra_info=dict(stage1_seconds=round(t1, 1), stage1_changed_frac=float(_up(ch_c, k)[region].mean()),
                                     band_frac=float(band[region].mean()), band_strands=int(in_band.sum()),
                                     n1=n1, res1=res1, width=width))
    fr.info["total_seconds"] = round(time.time() - t0, 1)
    return fr


VARIANTS = {
    # name: (n1 strands per head, res1, band width W texels)
    "ts_5k_r64_w8": (5000, 64, 8),
    "ts_5k_r64_w16": (5000, 64, 16),
    "ts_2k_r32_w16": (2000, 32, 16),
    # + coarse-to-fine membership (fast_membership.py; identical result, ~5x faster)
    "ts_5k_r64_w16_fast": (5000, 64, 16, 2.0, 0.7, True),
    "ts_2k_r32_w16_fast": (2000, 32, 16, 2.0, 0.7, True),
}


def _make(name, cfg):
    def method(heads, labels, ctx):
        return run_two_stage(heads, labels, ctx, *cfg)
    method.__name__ = method.__qualname__ = name
    method.__doc__ = f"two-stage FM {name}: n1, res1, width = {cfg}"
    register_method(name, supported_m=(2,))(method)


for _n, _c in VARIANTS.items():
    _make(_n, _c)
