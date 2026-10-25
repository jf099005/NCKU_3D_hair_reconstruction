"""Fusion-mask generators of the benchmark.

A generator draws a label map for m heads over the scalp-texture layout:

    @register_mask("name")
    def gen(rng, region, m, **kwargs) -> (labels, params)

    rng     np.random.Generator (the only source of randomness, so a case is reproducible from its seed)
    region  (res, res) bool, texels that carry hair roots of the case's heads (common.hair_region)
    m       number of heads
    labels  (res, res) int, 0 = head 0 (base), k = head k
    params  JSON-serializable dict of what was drawn (cut positions, polygon vertices, ...)

generate_labels() wraps a generator with rejection sampling so that every head gets an area
fraction of the hair region inside [min_frac, max_frac]. To add a new (e.g. more complex) mask
shape, write one function and decorate it; build_benchmark.py picks it up by name. All generators
here work for any m >= 2 (m - 1 cuts / m - 1 polygons).

Directions in the scalp-texture layout: columns run left <-> right on the head, rows front <-> back.
"""
import math

import cv2
import numpy as np

MASK_GENERATORS = {}


def register_mask(name):
    def deco(fn):
        MASK_GENERATORS[name] = fn
        return fn
    return deco


def _region_quantile_cuts(coord, region, m, rng, jitter):
    """m - 1 sorted cut positions along `coord` (per-texel coordinate map): equal-area quantiles of
    the region, each jittered by up to +-jitter of a part's share."""
    vals = np.sort(coord[region])
    qs = (np.arange(1, m) + rng.uniform(-jitter, jitter, m - 1)) / m
    qs = np.sort(np.clip(qs, 0.02, 0.98))
    return [float(vals[int(q * (vals.size - 1))]) for q in qs]


def _stripes(coord, region, m, rng, jitter):
    cuts = _region_quantile_cuts(coord, region, m, rng, jitter)
    stripe = np.searchsorted(np.asarray(cuts), coord, side="right")   # 0 .. m-1 along the axis
    order = rng.permutation(m)                                        # which head gets which stripe
    return order[stripe], cuts, order.tolist()


@register_mask("left_right")
def left_right(rng, region, m, jitter=0.6, **_):
    """Vertical cuts (texture columns = left <-> right on the head)."""
    res = region.shape[0]
    col = np.broadcast_to(np.arange(res, dtype=np.float32)[None, :], (res, res))
    labels, cuts, order = _stripes(col, region, m, rng, jitter)
    return labels, dict(axis="col", cuts=cuts, stripe_to_head=order)


@register_mask("front_back")
def front_back(rng, region, m, jitter=0.6, **_):
    """Horizontal cuts (texture rows = front <-> back of the head)."""
    res = region.shape[0]
    row = np.broadcast_to(np.arange(res, dtype=np.float32)[:, None], (res, res))
    labels, cuts, order = _stripes(row, region, m, rng, jitter)
    return labels, dict(axis="row", cuts=cuts, stripe_to_head=order)


@register_mask("half_plane")
def half_plane(rng, region, m, jitter=0.6, **_):
    """Parallel straight cuts at a random angle (left_right / front_back are its 90 / 0 degree cases)."""
    res = region.shape[0]
    theta = float(rng.uniform(0, math.pi))
    r, c = np.mgrid[0:res, 0:res].astype(np.float32)
    coord = c * math.cos(theta) + r * math.sin(theta)
    labels, cuts, order = _stripes(coord, region, m, rng, jitter)
    return labels, dict(angle_deg=math.degrees(theta), cuts=cuts, stripe_to_head=order)


@register_mask("polygon")
def polygon(rng, region, m, n_vertices=(3, 7), scale=(0.45, 0.95), **_):
    """Head k (k >= 1) gets a random convex polygon (later polygons drawn on top); head 0 keeps the rest.
    Each polygon: random center on a region texel, vertices at sorted random angles on an ellipse with
    radii = scale x the region's bounding box size, radius jittered per vertex."""
    res = region.shape[0]
    rows, cols = np.nonzero(region)
    extent = np.array([rows.max() - rows.min() + 1, cols.max() - cols.min() + 1], np.float32)
    labels = np.zeros((res, res), np.int16)
    polys = []
    for k in range(1, m):
        i = rng.integers(rows.size)
        center = np.array([rows[i], cols[i]], np.float32)
        nv = int(rng.integers(n_vertices[0], n_vertices[1] + 1))
        ang = np.sort(rng.uniform(0, 2 * math.pi, nv))
        radii = extent * rng.uniform(*scale, size=2) / 2
        jit = rng.uniform(0.75, 1.0, nv)
        pts_rc = center + np.stack([np.sin(ang), np.cos(ang)], 1) * radii * jit[:, None]
        hull = cv2.convexHull(np.round(pts_rc[:, ::-1]).astype(np.int32))   # cv2 wants (x=col, y=row)
        poly = np.zeros((res, res), np.uint8)
        cv2.fillPoly(poly, [hull], 1)
        labels[poly.astype(bool)] = k
        polys.append(dict(head=k, center_rc=center.tolist(), vertices_xy=hull.reshape(-1, 2).tolist()))
    return labels, dict(polygons=polys)


def area_fractions(labels, region, m):
    n = max(int(region.sum()), 1)
    return [float((labels[region] == k).sum()) / n for k in range(m)]


def generate_labels(name, rng, region, m, min_frac=0.15, max_frac=None, max_tries=200, **kwargs):
    """Draw one label map with generator `name`, redrawing until every head's share of `region` is in
    [min_frac, max_frac] (max_frac default 1 - (m - 1) * min_frac). Returns (labels, params) with
    params["area_fractions"] and params["tries"] added."""
    if name not in MASK_GENERATORS:
        raise KeyError(f"unknown mask generator {name!r}, known: {sorted(MASK_GENERATORS)}")
    if max_frac is None:
        max_frac = 1.0 - (m - 1) * min_frac
    for t in range(1, max_tries + 1):
        labels, params = MASK_GENERATORS[name](rng, region, m, **kwargs)
        fr = area_fractions(labels, region, m)
        if all(min_frac <= f <= max_frac for f in fr):
            params = dict(params, area_fractions=fr, tries=t)
            return labels.astype(np.int16), params
    raise RuntimeError(f"mask generator {name!r}: no label map with head area fractions in "
                       f"[{min_frac}, {max_frac}] after {max_tries} tries")
