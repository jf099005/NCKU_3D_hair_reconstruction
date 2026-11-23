"""Hair-flow field on the scalp texture, flow-aligned boundary band, LIC smoothing along the flow
(direction A of benchmark/LOG.md [10]-[12]).

flow_field: every strand's growth direction near the root (root -> point FLOW_ARC m along it, both mapped to
texels via the nearest root of either head), a sign-free structure tensor on a FLOW_GRID grid, smoothed ->
principal direction f ((row, col) components) and coherence c in [0, 1].
flow_band: Dijkstra from the drawn boundary on the 8-connected texel grid of the hair region; a step d (unit u)
costs |d| sqrt(k (u.f)^2 + (u.f_perp)^2), k = 1 - c (1 - 1/a^2); band = cost <= budget texels; cap limits the
band to that share of each head's region.
flow_smooth: Gaussian-weighted average along the streamline, coherence-blended, small blur, threshold 0.5.
"""
import numpy as np

from ._util import TEXTURE_RES, root_texels

FLOW_ARC = 0.015       # m along the strand for the growth direction
FLOW_GRID = 64         # structure-tensor grid (texture res / 4)
FLOW_SIGMA = 1.5       # smoothing, in FLOW_GRID cells


def flow_field(heads, res=TEXTURE_RES, step=2):
    """(f (res, res, 2) unit direction in (row, col), coherence (res, res))"""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    roots = np.concatenate([h["positions"][:, 0] for h in heads]).astype(np.float64)
    rc = np.concatenate([np.stack(root_texels(h["root_uv"], res), 1) for h in heads]).astype(np.float64)
    kd = cKDTree(roots)
    g = FLOW_GRID
    T = np.zeros((g, g, 3))
    for h in heads:
        P = h["positions"][::step].astype(np.float64)
        seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
        s = np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, 1)], 1)
        j = np.clip((s < FLOW_ARC).sum(1), 1, P.shape[1] - 1)
        tip = P[np.arange(len(P)), j]
        _, q = kd.query(tip)
        r0 = np.stack(root_texels(h["root_uv"][::step], res), 1).astype(np.float64)
        d = rc[q] - r0
        n = np.linalg.norm(d, axis=1)
        ok = n > 0.5
        d = d[ok] / n[ok, None]
        cell = np.clip((r0[ok] * g / res).astype(int), 0, g - 1)
        for k, v in enumerate((d[:, 0] ** 2, d[:, 0] * d[:, 1], d[:, 1] ** 2)):
            np.add.at(T[..., k], (cell[:, 0], cell[:, 1]), v)
    T = np.stack([gaussian_filter(T[..., k], FLOW_SIGMA) for k in range(3)], -1)
    a, b, c = T[..., 0], T[..., 1], T[..., 2]
    tr = a + c
    disc = np.sqrt(np.maximum((a - c) ** 2 / 4 + b ** 2, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    coh = np.where(tr > 1e-9, (l1 - l2) / np.maximum(l1 + l2, 1e-9), 0)
    ang = 0.5 * np.arctan2(2 * b, a - c)
    f = np.stack([np.cos(ang), np.sin(ang)], -1)          # (row, col) components
    up = lambda x: np.repeat(np.repeat(x, res // g, 0), res // g, 1)  # noqa: E731
    return np.stack([up(f[..., 0]), up(f[..., 1])], -1), up(coh)


def flow_band(labels, region, f, coh, budget, aniso, cap=None):
    """(res, res) bool: hair-region texels within `budget` (texels) of the drawn boundary, distances measured
    with the flow-aligned metric (module docstring)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    res = labels.shape[0]
    idx = -np.ones((res, res), np.int64)
    rr, cc = np.nonzero(region)
    idx[rr, cc] = np.arange(rr.size)
    k = 1 - coh * (1 - 1 / aniso ** 2)
    src, dst, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < res) & (c2 >= 0) & (c2 < res)
        ok[ok] &= region[r2[ok], c2[ok]]
        a, b = idx[rr[ok], cc[ok]], idx[r2[ok], c2[ok]]
        u = np.array([dr, dc], float) / np.hypot(dr, dc)
        fa = 0.5 * (f[rr[ok], cc[ok]] + f[r2[ok], c2[ok]])
        fa /= np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-9)
        along = (fa @ u) ** 2
        kk = 0.5 * (k[rr[ok], cc[ok]] + k[r2[ok], c2[ok]])
        cost = np.hypot(dr, dc) * np.sqrt(kk * along + (1 - along))
        src += [a, b]
        dst += [b, a]
        w += [cost, cost]
    G = coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(rr.size, rr.size)).tocsr()
    lab = labels > 0
    bnd = np.zeros_like(lab)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        sh = np.roll(np.roll(lab, dr, 0), dc, 1)
        shr = np.roll(np.roll(region, dr, 0), dc, 1)
        bnd |= region & shr & (sh != lab)
    srcs = idx[bnd]
    if srcs.size == 0:
        return np.zeros_like(region)
    dist = dijkstra(G, directed=True, indices=srcs, min_only=True, limit=budget + 1e-6)
    band = np.zeros_like(region)
    band[rr, cc] = np.isfinite(dist) & (dist <= budget)
    if cap is not None:   # the band may cover at most `cap` of each head's region: shrink that side's budget
        side = lab[rr, cc]
        for v in (False, True):
            on = side == v
            n_side = int(on.sum())
            d = np.where(np.isfinite(dist), dist, np.inf)[on]
            if n_side and (d <= budget).sum() > cap * n_side:
                lim = np.sort(d)[max(int(cap * n_side) - 1, 0)]
                sel = np.flatnonzero(on)
                band[rr[sel], cc[sel]] = d <= lim
    return band


def flow_smooth(mask, region, f, coh, sel, L=12, iso_sigma=1.0):
    """LIC-style smoothing along the hair flow: every texel of `sel` takes the Gaussian-weighted (sigma L/2)
    average of `mask` along its streamline (+-L unit steps following f, sign kept consistent, only hair-region
    samples count), blended with its own value by the coherence, then a small isotropic blur and threshold 0.5.
    Texels outside `sel` keep `mask`. -> (res, res) bool"""
    from scipy.ndimage import gaussian_filter
    res = mask.shape[0]
    m = mask.astype(np.float64)
    rr, cc = np.nonzero(region)
    acc = m[rr, cc].copy()
    wsum = np.ones(rr.size)
    for sgn in (1.0, -1.0):
        p = np.stack([rr, cc], 1).astype(np.float64)
        d = f[rr, cc] * sgn
        alive = np.ones(rr.size, bool)
        for t in range(1, L + 1):
            q = np.clip(np.rint(p).astype(int), 0, res - 1)
            fd = f[q[:, 0], q[:, 1]]
            fd = np.where(((fd * d).sum(1) < 0)[:, None], -fd, fd)
            d = fd
            p = p + d
            q = np.clip(np.rint(p).astype(int), 0, res - 1)
            alive &= region[q[:, 0], q[:, 1]]
            w = np.exp(-0.5 * (t / (0.5 * L)) ** 2) * alive
            acc += w * m[q[:, 0], q[:, 1]]
            wsum += w
    lic = np.zeros_like(m)
    lic[rr, cc] = acc / wsum
    c = coh
    out = c * lic + (1 - c) * m
    num = gaussian_filter(np.where(region, out, 0.0), iso_sigma)
    den = gaussian_filter(region.astype(np.float64), iso_sigma)
    sm = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0) > 0.5
    return np.where(sel & region, sm, mask.astype(bool))
