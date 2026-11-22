"""Metrics of a fused hairstyle, the same for every fusion method.

    @register_metric("name")
    def metric(cc, keep, init_keep) -> (scalars: dict, maps: dict)

    cc         CaseContext: the case's heads + an n x n analysis ScalpGrid (update_contour) and every
               head's dense strand/column membership on it, computed once per case and shared by all
               methods and metrics
    keep       per head (S_i,) bool, the method's result; init_keep = the drawn mask's (mask_only's)
    scalars    JSON-serializable numbers -> result.json / summary.csv (as "<metric>.<key>")
    maps       optional (n, n) arrays -> maps.npz (as "<metric>/<key>")

Grid maps are in update_contour's scalp-grid (row, col) layout. Pairwise quantities (crossings, grid
score) are summed / averaged over all head pairs, so they stay defined for m > 2.

Metrics:
  strands       kept strands per head, share of the fused hair from the donors, fraction of strands
                whose kept / dropped state differs from the drawn mask
  coverage      bald cells: scalp cells (n x n, scalp-texture layout) with roots of some input head
                but no kept root (holes the fusion opened)
  crossings     column rule of joint_contour: per analysis cell, kept strands of head i x kept strands of
                head j; conflict_cells = cells crossed by kept strands of both heads
  grid_score    convex-hull overlap score (mask_analysis.cal_grid_score, the score of
                img2hair_tsai_1.debug_field_analysis' fused_hull_score.png; larger = more mixed).
                Besides the average over all cells: the total score divided by the black (donor) and the
                white (base) area, the cells of the fused grid mask (update_contour.strand_mask_to_grid_mask
                = fused_grid_mask.png) where the donor's kept strands are a strict majority / the other
                cells crossed by kept strands
  flux          |3D flux| (divergence of the flow field, update_contour.cal_grid_flux_3d) of the fused hair
"""
import itertools
import time

import numpy as np

import common

METRICS = {}
DEFAULT_METRICS = ("strands", "coverage", "crossings", "grid_score", "flux")
GRID_SCORE_THRESH = 0.1


def register_metric(name):
    def deco(fn):
        METRICS[name] = fn
        return fn
    return deco


class CaseContext:
    def __init__(self, heads, grid_n, dataset_path, grid_cache):
        import highlighting.generate_from_3D_models.update_contour as uc
        self.uc = uc
        self.heads = heads
        self.n = grid_n
        key = ("analysis_grid", grid_n)
        if key not in grid_cache:
            grid_cache[key] = uc.ScalpGrid.from_dataset(grid_n, dataset_path)
        self.grid = grid_cache[key]
        self.arrays = self.grid.arrays()
        self.valid_rc = self.grid.valid_rc
        t0 = time.time()
        # (n, n, S_i) bool per head, (u_idx, v_idx) layout
        self.smg = [self.grid.strand_mask_grid(h["positions"], show_progress=False) for h in heads]
        self.membership_seconds = round(time.time() - t0, 2)

    def kept_counts_rc(self, keep):
        """per head (n, n) number of kept strands through each cell, (row, col) layout"""
        return [self.uc.uv_to_rc(smg[..., k].sum(-1)).astype(np.int64) for smg, k in zip(self.smg, keep)]

    def fused(self, keep, heads_idx=None):
        """positions, membership, head index of the fused hair made of the kept strands of heads_idx"""
        idx = range(len(self.heads)) if heads_idx is None else heads_idx
        pos = np.concatenate([self.heads[i]["positions"][keep[i]] for i in idx])
        smg = np.concatenate([self.smg[i][..., keep[i]] for i in idx], axis=-1)
        src = np.concatenate([np.full(int(keep[i].sum()), i, np.int16) for i in idx])
        return pos, smg, src


def _map_stats(x, valid, prefix=""):
    v = x[valid]
    if v.size == 0:
        return {f"{prefix}mean": 0.0}
    return {f"{prefix}mean": float(v.mean()), f"{prefix}p95": float(np.percentile(v, 95)),
            f"{prefix}max": float(v.max())}


@register_metric("strands")
def strands(cc, keep, init_keep):
    n = [int(k.size) for k in keep]
    kept = [int(k.sum()) for k in keep]
    total = max(sum(kept), 1)
    out = dict(n=n, kept=kept, fused_total=sum(kept), donor_share=float(sum(kept[1:]) / total),
               changed_frac=float(sum(int((k != k0).sum()) for k, k0 in zip(keep, init_keep)) / max(sum(n), 1)))
    return out, {}


@register_metric("coverage")
def coverage(cc, keep, init_keep):
    n = cc.n
    def occ(uv):
        r, c = common.root_texels(uv, n)
        o = np.zeros((n, n), bool)
        o[r, c] = True
        return o
    any_roots = np.zeros((n, n), bool)
    kept_roots = np.zeros((n, n), bool)
    for h, k in zip(cc.heads, keep):
        any_roots |= occ(h["root_uv"])
        kept_roots |= occ(h["root_uv"][k])
    bald = any_roots & ~kept_roots
    return dict(root_cells=int(any_roots.sum()), bald_cells=int(bald.sum()),
                bald_frac=float(bald.sum() / max(any_roots.sum(), 1))), {"bald": bald}


@register_metric("crossings")
def crossings(cc, keep, init_keep):
    counts = cc.kept_counts_rc(keep)
    pairs = np.zeros((cc.n, cc.n), np.int64)
    conflict = np.zeros((cc.n, cc.n), bool)
    for i, j in itertools.combinations(range(len(counts)), 2):
        pairs += counts[i] * counts[j]
        conflict |= (counts[i] > 0) & (counts[j] > 0)
    return dict(pairs=int(pairs.sum()), conflict_cells=int(conflict.sum()),
                conflict_frac=float(conflict.sum() / max(cc.valid_rc.sum(), 1))), {"pairs": pairs}


@register_metric("grid_score")
def grid_score(cc, keep, init_keep):
    import highlighting.generate_from_3D_models.mask_analysis as ma
    a = cc.arrays
    scores = []
    for i, j in itertools.combinations(range(len(cc.heads)), 2):
        pos, smg, src = cc.fused(keep, (i, j))
        scores.append(ma.cal_grid_score(pos, a["grid_pos"], a["grid_normal"], a["grid_valid"], src == j,
                                        a["half_size"], smg, grid_t1=a["grid_t1"], grid_t2=a["grid_t2"],
                                        show_progress=False))
    score = np.mean(scores, axis=0)
    counts = cc.kept_counts_rc(keep)
    # per pair (i = white, j = black): total score / area, then averaged over the pairs (m = 2: one pair)
    total, black, white, per_black, per_white = [], [], [], [], []
    for s, (i, j) in zip(scores, itertools.combinations(range(len(cc.heads)), 2)):
        b = counts[j] * 2 > counts[i] + counts[j]          # = strand_mask_to_grid_mask (ties -> white)
        w = (counts[i] + counts[j] > 0) & ~b
        total.append(float(s.sum()))
        black.append(int(b.sum()))
        white.append(int(w.sum()))
        per_black.append(total[-1] / max(black[-1], 1))
        per_white.append(total[-1] / max(white[-1], 1))
    mixed = sum((c > 0).astype(int) for c in counts) >= 2
    out = dict(mean_all=float(score.mean()),      # = img2hair_tsai_1 stats.json mean_hull_score
               **_map_stats(score, cc.valid_rc),
               **_map_stats(score, mixed, "mixed_"),
               frac_above=float((score[cc.valid_rc] > GRID_SCORE_THRESH).mean()),
               thresh=GRID_SCORE_THRESH,
               total=float(np.mean(total)), black_cells=float(np.mean(black)), white_cells=float(np.mean(white)),
               per_black_area=float(np.mean(per_black)), per_white_area=float(np.mean(per_white)))
    return out, {"score": score}


@register_metric("flux")
def flux(cc, keep, init_keep):
    a = cc.arrays
    pos, smg, _ = cc.fused(keep)
    f3, ok = cc.uc.cal_grid_flux_3d(pos, a["grid_pos"], a["grid_normal"], a["grid_valid"], a["half_size"], smg,
                                    grid_t1=a["grid_t1"], grid_t2=a["grid_t2"], show_progress=False)
    absf = np.abs(f3).astype(np.float32)
    return dict(**_map_stats(absf, ok & cc.valid_rc, "abs_")), {"abs_flux": absf}


def evaluate(cc, keep, init_keep, names=DEFAULT_METRICS):
    scalars, maps, timings = {}, {}, {}
    for name in names:
        t0 = time.time()
        s, mp = METRICS[name](cc, keep, init_keep)
        timings[name] = round(time.time() - t0, 2)
        scalars[name] = s
        maps.update({f"{name}/{k}": v for k, v in mp.items()})
    return scalars, maps, timings


# ---------------------------------------------------------------- view_mix (opt-in: --metrics ...,view_mix)
# What the renders show, without Blender: every kept strand point is splatted into orthographic z-buffers of
# VIEW_DIRS (front, back, left, right, top); each pixel shows the head of its nearest point. Unlike the column
# grid score this sees layering (a strand lifted over the other head hides it), so it can score strand editing.
#   edge_frac   4-neighbor pixel pairs (both hair) showing different heads / hair pixels (speckle + boundary)
#   block_mix   mean over VIEW_BLOCK x VIEW_BLOCK blocks of 4 p (1 - p), p = donor share of its hair pixels,
#               weighted by hair pixels (0 = every block one head; 1 = 50/50 salt and pepper)
#   donor_vis   donor share of all visible hair pixels (compare with strands.donor_share: a head that vanishes
#               from view shows up here)
VIEW_DIRS = {"front": (0, 0, 1), "back": (0, 0, -1), "left": (1, 0, 0), "right": (-1, 0, 0), "top": (0, 1, 0)}
VIEW_RES, VIEW_HALF, VIEW_BLOCK, VIEW_STEP = 192, 0.22, 8, 2


def _view_frame(d):
    d = np.asarray(d, np.float64)
    d /= np.linalg.norm(d)
    up = np.array([0, 1.0, 0]) if abs(d[1]) < 0.9 else np.array([0, 0, -1.0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x)
    return x, np.cross(d, x), d


def view_images(cc, keep):
    """{view: (VIEW_RES, VIEW_RES) int8 image, -1 = no hair, i = head i}"""
    roots = np.concatenate([h["positions"][:, 0] for h in cc.heads])
    c = roots.mean(0)
    pts = np.concatenate([h["positions"][k][:, ::VIEW_STEP].reshape(-1, 3) for h, k in zip(cc.heads, keep)])
    src = np.concatenate([np.full(int(k.sum()) * h["positions"][:, ::VIEW_STEP].shape[1], i, np.int8)
                          for i, (h, k) in enumerate(zip(cc.heads, keep))])
    out = {}
    for name, d in VIEW_DIRS.items():
        x, y, z = _view_frame(d)
        q = pts - c
        u = np.floor((q @ x + VIEW_HALF) / (2 * VIEW_HALF) * VIEW_RES).astype(np.int64)
        v = np.floor((VIEW_HALF - q @ y) / (2 * VIEW_HALF) * VIEW_RES).astype(np.int64)
        depth = -(q @ z)                      # smaller = closer to the camera (camera at +d)
        ok = (u >= 0) & (u < VIEW_RES) & (v >= 0) & (v < VIEW_RES)
        pix = (v * VIEW_RES + u)[ok]
        order = np.lexsort((depth[ok], pix))
        pix_o = pix[order]
        first = np.ones(pix_o.size, bool)
        first[1:] = pix_o[1:] != pix_o[:-1]
        img = np.full(VIEW_RES * VIEW_RES, -1, np.int8)
        img[pix_o[first]] = src[ok][order][first]
        out[name] = img.reshape(VIEW_RES, VIEW_RES)
    return out


@register_metric("view_mix")
def view_mix(cc, keep, init_keep):
    imgs = view_images(cc, keep)
    edge, blk, dv, maps = [], [], [], {}
    for name, img in imgs.items():
        hair = img >= 0
        n_h = max(int(hair.sum()), 1)
        d = 0
        for a, b in ((img[1:], img[:-1]), (img[:, 1:], img[:, :-1])):
            d += int(((a >= 0) & (b >= 0) & (a != b)).sum())
        edge.append(d / n_h)
        B = VIEW_BLOCK
        h_ = hair.reshape(VIEW_RES // B, B, VIEW_RES // B, B).sum((1, 3)).astype(np.float64)
        don = (img > 0).reshape(VIEW_RES // B, B, VIEW_RES // B, B).sum((1, 3))
        p = np.where(h_ > 0, don / np.maximum(h_, 1), 0)
        blk.append(float((4 * p * (1 - p) * h_).sum() / max(h_.sum(), 1)))
        dv.append(float((img > 0).sum() / n_h))
        maps[name] = img
    return (dict(edge_frac=float(np.mean(edge)), block_mix=float(np.mean(blk)), donor_vis=float(np.mean(dv)),
                 **{f"edge_frac_{k}": float(e) for k, e in zip(imgs, edge)}), maps)
