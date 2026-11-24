"""Scalp-texture features of a two-head fusion case, and the teacher targets / their application.

Everything is on a (R, R) grid in the scalp-texture layout (R = FEAT_RES by default; the 256 masks are pooled).
Vector quantities are in a per-texel local frame (t_col, t_row, n): t_col along increasing column, t_row along
increasing row, n = outward (radial from the root centroid). A left-right flip of the texture (col -> R-1-col)
is then the mirror image when the t_col components are negated (FLIP_SIGN_* say which channels).

features(heads, labels)            -> dict(x (C, R, R) float32, region (R, R) bool, frame ...)
targets(heads, keep, mask, edited) -> (1 + 3K, R, R): refined donor fraction, mean displacement at K knots
apply(heads, keep, field, frame)   -> edited heads (root fixed, follow-the-leader lengths, kink limit)
"""
import numpy as np

from fusion_refine._util import hair_region, root_texels

FEAT_RES = 64
PT_STEP = 8                        # strand point subsampling for footprints (256 -> 32 points)
SHAPE_IDX = (16, 48, 112, 255)     # strand points whose (point - root) describe the shape per root texel
KNOT_IDX = (16, 40, 72, 112, 152, 192, 224, 255)   # displacement knots
HEIGHT_BINS = (0.01, 0.03)         # m, height above the scalp of footprint points: <1, 1-3, >3 cm
VOX = 0.003                        # m, mixing test voxel (as comb.py)
SHAPE_SCALE = 0.1                  # m
DISP_SCALE = 0.005                 # m; teacher moves are sparse, median ~2 mm, max ~2 cm


def _fill_nearest(val, has):
    from scipy.ndimage import distance_transform_edt
    _, (ir, ic) = distance_transform_edt(~has, return_indices=True)
    return val[ir, ic]


def scalp_frame(heads, R=FEAT_RES):
    """per texel scalp point (R, R, 3) and frame (R, R, 3, 3) rows = (t_col, t_row, n)"""
    acc = np.zeros((R, R, 3))
    cnt = np.zeros((R, R))
    for h in heads:
        r, c = root_texels(h["root_uv"], R)
        np.add.at(acc, (r, c), h["positions"][:, 0].astype(np.float64))
        np.add.at(cnt, (r, c), 1)
    has = cnt > 0
    acc[has] /= cnt[has, None]
    tex3d = _fill_nearest(acc, has)
    from scipy.ndimage import gaussian_filter
    sm = np.stack([gaussian_filter(tex3d[..., k], 1.5, mode="nearest") for k in range(3)], -1)
    center = np.concatenate([h["positions"][:, 0] for h in heads]).mean(0)
    n = sm - center
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    tc = np.gradient(sm, axis=1)
    tc -= (tc * n).sum(-1, keepdims=True) * n
    tc /= np.maximum(np.linalg.norm(tc, axis=-1, keepdims=True), 1e-9)
    tr = np.cross(n, tc)          # n x t_col; sign fixed so that it points along increasing row
    sgn = np.sign((tr * np.gradient(sm, axis=0)).sum(-1, keepdims=True))
    sgn[sgn == 0] = 1
    tr *= sgn
    return dict(tex3d=tex3d, frame=np.stack([tc, tr, n], -2), center=center, R=R)


def _to_local(vec, frame_at):
    """vec (..., 3) world, frame_at (..., 3, 3) -> local (..., 3)"""
    return np.einsum("...ij,...j->...i", frame_at, vec)


def _footprints(heads, R, step=PT_STEP):
    """per head: points (S, J, 3) subsampled, footprint texel (S*J,) rows / cols at R (nearest root of any head)"""
    from scipy.spatial import cKDTree
    roots = np.concatenate([h["positions"][:, 0] for h in heads]).astype(np.float64)
    rc = np.concatenate([np.stack(root_texels(h["root_uv"], R), 1) for h in heads])
    kd = cKDTree(roots)
    out = []
    for h in heads:
        X = h["positions"][:, ::step].astype(np.float64)
        _, q = kd.query(X.reshape(-1, 3), workers=-1)
        out.append((X, rc[q, 0], rc[q, 1]))
    return out


def _voxel_occupancy(point_sets, vox=VOX):
    from scipy.ndimage import uniform_filter
    allP = np.concatenate([p.reshape(-1, 3) for p in point_sets])
    lo = allP.min(0) - 2 * vox
    dims = np.ceil((allP.max(0) + 2 * vox - lo) / vox).astype(int) + 1
    occ = []
    for p in point_sets:
        g = np.zeros(dims, np.float32)
        v = np.floor((p.reshape(-1, 3) - lo) / vox).astype(int)
        np.add.at(g, tuple(v.T), 1)
        occ.append(uniform_filter(g, 3, mode="constant") * 27)
    return occ, lo, dims


CHANNELS = []   # filled below: names, for documentation / flip signs


def features(heads, labels, R=FEAT_RES, keep=None):
    """labels: (256, 256) drawn label map (1 = donor) in the scalp-texture layout. keep: optional per head (S,)
    bool -- the kept strands the footprint / mixing features are computed on (default: the drawn split; that is
    what the model sees at inference)."""
    from fusion_refine._util import initial_keep
    fr = scalp_frame(heads, R)
    frame = fr["frame"]
    region256 = hair_region([h["root_uv"] for h in heads], labels.shape[0])
    k = labels.shape[0] // R
    lab = (labels == 1).reshape(R, k, R, k).mean((1, 3)).astype(np.float32)
    region = region256.reshape(R, k, R, k).mean((1, 3)) > 0.25
    keep = initial_keep(heads, labels) if keep is None else keep
    S_tot = sum(len(h["root_uv"]) for h in heads)
    per_texel = S_tot / max(region.sum(), 1) / 2       # expected strands of one head per texel
    ch = [lab, region.astype(np.float32)]
    rr, cc = np.meshgrid(np.linspace(-1, 1, R), np.linspace(-1, 1, R), indexing="ij")
    ch += [rr.astype(np.float32), cc.astype(np.float32)]
    # root-based: density, shape (point - root in the root texel's frame), length
    for h in heads:
        r, c = root_texels(h["root_uv"], R)
        cnt = np.zeros((R, R))
        np.add.at(cnt, (r, c), 1)
        ch.append((cnt / per_texel).astype(np.float32))
        P = h["positions"]
        loc = _to_local(P[:, list(SHAPE_IDX)] - P[:, :1], frame[r, c][:, None])     # (S, 4, 3)
        sh = np.zeros((R, R, len(SHAPE_IDX), 3))
        np.add.at(sh, (r, c), loc)
        L = np.linalg.norm(np.diff(P[:, ::4], axis=1), axis=-1).sum(1)
        ln = np.zeros((R, R))
        np.add.at(ln, (r, c), L)
        m = np.maximum(cnt, 1)
        ch += list((sh / m[..., None, None] / SHAPE_SCALE).reshape(R, R, -1).transpose(2, 0, 1).astype(np.float32))
        ch.append((ln / m / SHAPE_SCALE).astype(np.float32))
    # footprint-based (kept strands only): height bins, mean direction, mixing with the other head
    fps = _footprints(heads, R)
    pts = [fp[0][kp] for fp, kp in zip(fps, keep)]
    occ, lo, dims = _voxel_occupancy(pts)
    for i, ((X, fr_, fc_), kp) in enumerate(zip(fps, keep)):
        S, J, _ = X.shape
        sel = np.repeat(kp, J)
        frr, fcc = fr_[sel], fc_[sel]
        Xs = X[kp].reshape(-1, 3)
        hgt = ((Xs - fr["tex3d"][frr, fcc]) * frame[frr, fcc, 2]).sum(-1)
        b = np.digitize(hgt, HEIGHT_BINS)
        norm = per_texel * J
        for bi in range(len(HEIGHT_BINS) + 1):
            g = np.zeros((R, R))
            np.add.at(g, (frr[b == bi], fcc[b == bi]), 1)
            ch.append((g / norm).astype(np.float32))
        d = np.gradient(X[kp], axis=1).reshape(-1, 3)
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        dl = _to_local(d, frame[frr, fcc])
        g = np.zeros((R, R, 3))
        np.add.at(g, (frr, fcc), dl)
        cnt = np.zeros((R, R))
        np.add.at(cnt, (frr, fcc), 1)
        ch += list((g / np.maximum(cnt, 1)[..., None]).transpose(2, 0, 1).astype(np.float32))
        v = np.clip(np.floor((Xs - lo) / VOX).astype(int), 0, dims - 1)
        mix = occ[1 - i][tuple(v.T)] >= 2
        g = np.zeros((R, R))
        np.add.at(g, (frr[mix], fcc[mix]), 1)
        ch.append((g / norm).astype(np.float32))
    x = np.stack(ch)
    x[:, ~region] = 0
    x[0] = lab   # keep the drawn label everywhere
    return dict(x=x, region=region, frame=frame, tex3d=fr["tex3d"], R=R)


def flip_signs():
    """(C,) multiplier of each feature channel under a left-right flip (t_col components and the col coordinate)"""
    s = [1, 1, 1, -1]                                    # lab, region, row, col
    for _ in range(2):
        s += [1] + [-1, 1, 1] * len(SHAPE_IDX) + [1]      # density, shape, length
    for _ in range(2):
        s += [1] * (len(HEIGHT_BINS) + 1) + [-1, 1, 1] + [1]
    return np.array(s, np.float32)


def target_flip_signs():
    return np.array([1] + [-1, 1, 1] * len(KNOT_IDX), np.float32)


N_FEAT = len(flip_signs())
N_TARGET = 1 + 3 * len(KNOT_IDX)


def targets(heads, keep, mask, edited, fe):
    """mask: (256, 256) refined (True = donor); edited: per head positions after the teacher's displacement.
    -> (1 + 3K, R, R) float32: donor fraction in [0, 1] (as 2f - 1), mean displacement of the kept strands at the
    knots in the root texel's frame / DISP_SCALE; and the per-texel weight (R, R) of kept strands."""
    R = fe["R"]
    k = mask.shape[0] // R
    y = np.zeros((N_TARGET, R, R), np.float32)
    y[0] = mask.reshape(R, k, R, k).mean((1, 3)) * 2 - 1
    acc = np.zeros((R, R, len(KNOT_IDX), 3))
    cnt = np.zeros((R, R))
    for h, kp, E in zip(heads, keep, edited):
        r, c = root_texels(h["root_uv"][kp], R)
        D = (np.asarray(E[kp], np.float64) - h["positions"][kp])[:, list(KNOT_IDX)]
        np.add.at(acc, (r, c), _to_local(D, fe["frame"][r, c][:, None]))
        np.add.at(cnt, (r, c), 1)
    y[1:] = (acc / np.maximum(cnt, 1)[..., None, None] / DISP_SCALE).reshape(R, R, -1).transpose(2, 0, 1)
    return y, cnt.astype(np.float32)


def mask_from_field(y0, region, res=256, thr=0.0):
    """donor channel (R, R) in [-1, 1] -> (res, res) bool mask (nearest upsampling)"""
    R = y0.shape[0]
    m = (y0 > thr) & region
    return np.repeat(np.repeat(m, res // R, 0), res // R, 1)


def apply(heads, keep, disp, fe, max_kink=30.0, d_max=None, disp_scale=DISP_SCALE):
    """disp: (3K, R, R) displacement field (/ DISP_SCALE, local frame). Kept strands move by the field sampled
    (bilinear) at their root, interpolated along the strand, ramped in from the root (smoothstep over 15 % of the
    length), then follow-the-leader restores segment lengths, the root stays fixed and the turning-angle increase
    is limited (as comb.py). -> edited heads, info"""
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss
    from scipy.ndimage import map_coordinates
    R = fe["R"]
    K = len(KNOT_IDX)
    field = disp.reshape(K, 3, R, R) * disp_scale
    out, info = [], {}
    for i, (h, kp) in enumerate(zip(heads, keep)):
        idx = np.flatnonzero(kp)
        pos = h["positions"].copy()
        if idx.size == 0:
            out.append(dict(h, positions=pos))
            continue
        uv = h["root_uv"][idx] * R - 0.5
        coords = [np.clip(uv[:, 0], 0, R - 1), np.clip(uv[:, 1], 0, R - 1)]
        loc = np.stack([[map_coordinates(field[kk, j], coords, order=1, mode="nearest") for j in range(3)]
                        for kk in range(K)])                       # (K, 3, S)
        r, c = root_texels(h["root_uv"][idx], R)
        world = np.einsum("sji,ksj->ski", fe["frame"][r, c], loc.transpose(0, 2, 1))    # frame^T @ local
        if d_max is not None:
            nrm = np.linalg.norm(world, axis=-1, keepdims=True)
            world *= np.minimum(1.0, d_max / np.maximum(nrm, 1e-12))
        P0 = h["positions"][idx].astype(np.float64)
        Pn = P0.shape[1]
        full = _interp_knots(world, np.array(KNOT_IDX, float), Pn)
        s, seg = dfm.arclength(P0)
        ramp = dfm.smoothstep(s / (0.15 * np.maximum(s[:, -1:], 1e-6)))
        moved = np.abs(full).max((1, 2)) > 1e-5
        Q = P0 + ramp[..., None] * full
        mi = np.flatnonzero(moved)
        nk = 0
        if mi.size:
            Qi = dfm.follow_the_leader(Q[mi], seg[mi])
            Qi[:, 0] = P0[mi, 0]
            Q[mi] = Qi
            Q, nk = ss.limit_kinks(P0, Q, mi, max_kink)
        bad = ~np.isfinite(Q).all((1, 2))
        Q[bad] = P0[bad]
        disp_mm = np.linalg.norm(Q - P0, axis=-1).max(1) * 1e3
        tag = "base" if i == 0 else "donor"
        info.update({f"{tag}_moved_strands": int(moved.sum()), f"{tag}_kink_limited": int(nk),
                     f"{tag}_disp_mean_mm": float(disp_mm[moved].mean()) if moved.any() else 0.0,
                     f"{tag}_disp_p99_mm": float(np.percentile(disp_mm, 99))})
        pos[idx] = Q.astype(pos.dtype)
        out.append(dict(h, positions=pos))
    return out, info


def _interp_knots(world, kn, Pn):
    """world (S, K, 3) displacements at point indices kn -> (S, Pn, 3), 0 at the root, linear in between"""
    xs = np.r_[0.0, kn]
    t = np.arange(Pn, dtype=float)
    j = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 2)
    w = ((t - xs[j]) / (xs[j + 1] - xs[j]))[None, :, None]
    W = np.concatenate([np.zeros_like(world[:, :1]), world], 1)
    return W[:, j] * (1 - w) + W[:, j + 1] * w


# ---------------------------------------------------------------- per-strand descriptors (keep decision)

STRAND_FEAT_NAMES = (["head", "drawn_own", "u", "v"] + [f"shape{k}_{c}" for k in range(len(SHAPE_IDX)) for c in "xyz"]
                     + ["length", "intr", "mix", "mix_intr"] + [f"mix_h{b}" for b in range(len(HEIGHT_BINS) + 1)]
                     + ["mix_near_root", "intr_near_root", "other_density_at_root", "dist_boundary"])


def strand_features(heads, labels, fe, keep=None):
    """per head (S, D) float32 descriptors of every strand (D = len(STRAND_FEAT_NAMES)):
    head id, root label == own head, root uv, shape (point - root, root frame, / SHAPE_SCALE), length, share of its
    points over the other head's drawn region (intr), mixing with the other head's kept hair (mix, as comb.py;
    keep = drawn split by default), mixing while intruding, mixing per height bin, the same near the root (first
    quarter), density of the other head at the root texel, distance of the root to the drawn boundary (texels/32)."""
    from fusion_refine._util import initial_keep
    from scipy.ndimage import distance_transform_edt
    R = fe["R"]
    frame = fe["frame"]
    res = labels.shape[0]
    lab = labels == 1
    region = hair_region([h["root_uv"] for h in heads], res)
    keep = initial_keep(heads, labels) if keep is None else keep
    dist_b = np.where(lab, distance_transform_edt(lab), distance_transform_edt(~lab)) / 32.0
    fps = _footprints(heads, res)                     # footprints at full res (for the drawn label)
    pts = [fp[0][kp] for fp, kp in zip(fps, keep)]
    occ, lo, dims = _voxel_occupancy(pts)
    dens = []
    for h in heads:
        r, c = root_texels(h["root_uv"], R)
        g = np.zeros((R, R))
        np.add.at(g, (r, c), 1)
        dens.append(g / max(g[fe["region"]].mean(), 1e-6))
    out = []
    for i, (h, (X, frr, fcc)) in enumerate(zip(heads, fps)):
        S, J, _ = X.shape
        P = h["positions"]
        r256, c256 = root_texels(h["root_uv"], res)
        rR, cR = root_texels(h["root_uv"], R)
        own = lab[r256, c256] == bool(i)
        loc = _to_local(P[:, list(SHAPE_IDX)] - P[:, :1], frame[rR, cR][:, None]).reshape(S, -1) / SHAPE_SCALE
        L = np.linalg.norm(np.diff(P[:, ::4], axis=1), axis=-1).sum(1) / SHAPE_SCALE
        intr = ((lab[frr, fcc] != bool(i)) & region[frr, fcc]).reshape(S, J)
        v = np.clip(np.floor((X.reshape(-1, 3) - lo) / VOX).astype(int), 0, dims - 1)
        mix = (occ[1 - i][tuple(v.T)] >= 2).reshape(S, J)
        fR, cRf = frr // (res // R), fcc // (res // R)
        hgt = ((X.reshape(-1, 3) - fe["tex3d"][fR, cRf]) * frame[fR, cRf, 2]).sum(-1).reshape(S, J)
        b = np.digitize(hgt, HEIGHT_BINS)
        q = max(1, J // 4)
        cols = [np.full(S, float(i)), own.astype(float), h["root_uv"][:, 0], h["root_uv"][:, 1]]
        cols += list(loc.T) + [L, intr.mean(1), mix.mean(1), (mix & intr).mean(1)]
        cols += [(mix & (b == bi)).mean(1) for bi in range(len(HEIGHT_BINS) + 1)]
        cols += [mix[:, :q].mean(1), intr[:, :q].mean(1), dens[1 - i][rR, cR], dist_b[r256, c256]]
        out.append(np.stack(cols, 1).astype(np.float32))
    return out
