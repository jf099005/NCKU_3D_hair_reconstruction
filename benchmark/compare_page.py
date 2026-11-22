"""Comparison images of a run's optical renders (render_results.py --optical), one grid per case:

    row 1  inputs:  original base | original donor | drawn mask 2D | drawn mask on the scalp (4 views) | legend
    row 2  each method's fused hair (optical render) + numbers
    row 3  each method's kept-strand mask on the scalp (4 views)
    row 4  each method's kept-strand mask in 2D (scalp-texture layout)

    python benchmark/compare_page.py <run_dir> [--cases a,b] [--methods m1,m2] [--device CUDA]
    -> <run_dir>/compare/compare_<case>.png  and  <run_dir>/compare/index.html

Colors follow the optical render (strands_editing.pipeline OPTICAL_COLORS): base = dark brown, donor = golden.
Kept-strand mask = what the method actually keeps: per BIN x BIN texel bin, the donor share of the kept roots
(brown -> gold), red where the inputs had roots but nothing is kept (bald), light gray outside the hair region.
The originals are each input hairstyle alone (all strands) through the same optical preset (cached in
<run_dir>/compare/<case>/original_{base,donor}/). The scalp views are a quick matplotlib orthographic render
of the body mesh (gray) with the scalp mesh (subdivided) colored by a texture at each face's texel
(scalp-texture layout: row = 1 - t, col = s; scalp.ply's (s, t) = the dataset root_uv). Views (front, top,
left, back) as the optical multiview.
"""
import argparse
import json
import os

import cv2
import numpy as np

import common

BASE_RGB = (0.36, 0.18, 0.08)      # dark brown
DONOR_RGB = (0.85, 0.66, 0.30)     # golden
OFF_RGB = (0.80, 0.80, 0.80)
BALD_RGB = (0.85, 0.12, 0.12)
BIN = 4
SKIN_RGB = (0.93, 0.88, 0.86)
VIEWS = {"front": ((0, 0, 1), (0, 1, 0)), "top": ((0, 1, 0), (0, 0, -1)),
         "left": ((-1, 0, 0), (0, 1, 0)), "back": ((0, 0, -1), (0, 1, 0))}
PANEL = 420


def _load_ply(path):
    import trimesh
    m = trimesh.load(path, process=False)
    raw = m.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
    uv = None
    if raw is not None and "s" in raw.dtype.names:
        uv = np.stack([raw["s"], raw["t"]], 1).astype(np.float64)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), uv


def _subdivide(V, F, UV, times):
    for _ in range(times):
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        mid = len(V) + np.arange(len(uniq))
        V = np.concatenate([V, 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])])
        UV = np.concatenate([UV, 0.5 * (UV[uniq[:, 0]] + UV[uniq[:, 1]])])
        n = len(F)
        a, b, c = mid[inv[:n]], mid[inv[n:2 * n]], mid[inv[2 * n:]]
        F = np.concatenate([np.stack([F[:, 0], a, c], 1), np.stack([a, F[:, 1], b], 1),
                            np.stack([c, b, F[:, 2]], 1), np.stack([a, b, c], 1)])
    return V, F, UV


def drawn_texture(labels, region):
    """(res, res, 3) RGB of the drawn mask"""
    tex = np.where((labels > 0)[..., None], DONOR_RGB, BASE_RGB).astype(np.float64)
    tex[~region] = OFF_RGB
    return tex


def kept_texture(root_uvs, keep, res=common.TEXTURE_RES, bin_=BIN):
    """(res, res, 3) RGB of the kept strands: donor share per bin (brown -> gold), bald bins red, no roots gray"""
    n = res // bin_
    cnt = np.zeros((2, n, n))
    inp = np.zeros((n, n))
    for i, (uv, k) in enumerate(zip(root_uvs, keep)):
        r, c = common.root_texels(uv, res)
        np.add.at(inp, (r // bin_, c // bin_), 1)
        np.add.at(cnt[min(i, 1)], (r[k] // bin_, c[k] // bin_), 1)
    tot = cnt.sum(0)
    f = np.where(tot > 0, cnt[1] / np.maximum(tot, 1), 0)[..., None]
    tex = (1 - f) * np.asarray(BASE_RGB) + f * np.asarray(DONOR_RGB)
    tex[(inp > 0) & (tot == 0)] = BALD_RGB
    tex[inp == 0] = OFF_RGB
    return np.repeat(np.repeat(tex, bin_, 0), bin_, 1)


_MESH = {}


def scalp_mask_views(tex, dataset_path, out_png):
    """4-view render of the head with the scalp colored by `tex` ((res, res, 3), scalp-texture layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    if dataset_path not in _MESH:
        body = os.path.join(dataset_path, "body_data")
        _MESH[dataset_path] = (_load_ply(os.path.join(body, "smplx_base.ply"))[:2],
                               _subdivide(*_load_ply(os.path.join(body, "scalp.ply")), 3))
    (Vb, Fb), (Vs, Fs, UVs) = _MESH[dataset_path]
    res = tex.shape[0]
    uvc = UVs[Fs].mean(1)
    r = np.clip(np.floor((1 - uvc[:, 1]) * res).astype(int), 0, res - 1)
    c = np.clip(np.floor(uvc[:, 0] * res).astype(int), 0, res - 1)
    col_s = tex[r, c]
    # keep the head only (vertices above the neck), push the scalp 1 mm outward against z-fighting
    keep_v = Vb[:, 1] > Vs[:, 1].min() - 0.12
    Fb = Fb[keep_v[Fb].all(1)]
    nrm_s = np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]])
    vn = np.zeros_like(Vs)
    for k in range(3):
        np.add.at(vn, Fs[:, k], nrm_s)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    out = Vs.mean(0)
    sgn = np.sign(((Vs - out) * vn).sum(1, keepdims=True).mean())
    Vs = Vs + 0.001 * sgn * vn
    center = Vs.mean(0)
    head_c = Vb[np.unique(Fb)].mean(0)

    def outward(V, F):
        T = V[F]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        flip = ((T.mean(1) - head_c) * n).sum(1) < 0
        n[flip] *= -1
        return T, n

    layers = [outward(Vb, Fb) + (np.tile(SKIN_RGB, (len(Fb), 1)),), outward(Vs, Fs) + (col_s,)]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=110)
    for ax, (name, (d, up)) in zip(axes.ravel(), VIEWS.items()):
        d, up = np.asarray(d, float), np.asarray(up, float)
        x = np.cross(up, d)
        for T, n, cols in layers:          # head first, scalp on top; back faces culled
            q = T - center
            px, py, pz = q @ x, q @ up, q @ d
            facing = n @ d
            order = np.argsort(pz.mean(1))
            order = order[facing[order] > 0]
            fc = np.clip(cols[order] * (0.5 + 0.5 * facing[order])[:, None], 0, 1)
            ax.add_collection(PolyCollection(np.stack([px[order], py[order]], -1), facecolors=fc, edgecolors=fc,
                                             linewidths=0.3))
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.13 if name != "top" else -0.12, 0.11 if name != "top" else 0.12)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=9)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.02, hspace=0.08)
    fig.savefig(out_png, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def mask_2d(tex, out_png, size=PANEL):
    img = (np.clip(tex, 0, 1)[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(out_png, cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST))


def original_renders(case, out_dir, device=None):
    """optical render of each input hairstyle alone (all strands): base colored as base, donor as donor"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    out = []
    for k, h in enumerate(case["heads"][:2]):
        rd = os.path.join(out_dir, "original_base" if k == 0 else "original_donor")
        png = os.path.join(rd, "front_multiview.png")
        if not os.path.exists(png):
            d = np.load(h["npz_path"])
            meta = json.load(open(os.path.join(os.path.dirname(h["npz_path"]), "metadata.json")))
            S = d["positions"].shape[0]
            print(f"  original {'base' if k == 0 else 'donor'} {h['name']}: optical render ({S} strands)")
            args = (rd, d["positions"].astype(np.float32), np.full(S, k > 0), d["root_uv"].astype(np.float32),
                    d["root_normal"].astype(np.float32))
            try:
                pl.render(*args, preset="optical", meta=meta, device=device)
            except RuntimeError:   # OptiX "Misaligned address" on some hair scenes (Blender 5.0): retry with CUDA
                print("    OptiX failed, retrying with CYCLES_DEVICE=CUDA")
                pl.render(*args, preset="optical", meta=meta, device="CUDA")
        out.append(png)
    return out


def _img(path, w, h):
    im = cv2.imread(path) if path and os.path.exists(path) else None
    if im is None:
        im = np.full((h, w, 3), 230, np.uint8)
        cv2.putText(im, "missing", (w // 3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)
        return im
    s = min(w / im.shape[1], h / im.shape[0])
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


def _text_block(lines, w, h, scale=0.55, bold_first=True):
    out = np.full((h, w, 3), 255, np.uint8)
    y = 24
    for i, ln in enumerate(lines):
        big = bold_first and i == 0
        cv2.putText(out, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68 if big else scale, (0, 0, 0) if big else (55, 55, 55),
                    2 if big else 1, cv2.LINE_AA)
        y += 28 if big else 22
    return out


def _cell(title_lines, path, w=PANEL, h=PANEL, head=64):
    return np.vstack([_text_block(title_lines, w, head), _img(path, w, h)])


def _legend(w=PANEL, h=PANEL + 64):
    out = np.full((h, w, 3), 255, np.uint8)
    items = [("base", BASE_RGB), ("donor", DONOR_RGB), ("mixed (donor share)", None),
             ("bald: roots in input, none kept", BALD_RGB), ("no roots", OFF_RGB)]
    cv2.putText(out, "legend", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 2, cv2.LINE_AA)
    y = 60
    for name, rgb in items:
        if rgb is None:
            for i in range(40):
                f = i / 39
                c = (1 - f) * np.asarray(BASE_RGB) + f * np.asarray(DONOR_RGB)
                out[y - 16:y + 4, 10 + i:11 + i] = (np.asarray(c)[::-1] * 255).astype(np.uint8)
        else:
            out[y - 16:y + 4, 10:50] = (np.asarray(rgb)[::-1] * 255).astype(np.uint8)
        cv2.putText(out, name, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
        y += 34
    notes = ["gs  = grid score (lower = less mixed)", "ret = min head retention vs the drawn mask",
             "view_mix = visible color-edge fraction", "fused hair: optical render,", "  front / top / left / back"]
    for ln in notes:
        cv2.putText(out, ln, (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
        y += 24
    return out


def _metrics(res_path, ref):
    if not os.path.exists(res_path):
        return ["(no result)"]
    j = common.read_json(res_path)
    if j.get("status") != "ok":
        return [j.get("status", "?")]
    m = j["metrics"]
    kept, k0 = m["strands"]["kept"], ref["metrics"]["strands"]["kept"]
    ret = min(k / max(b, 1) for k, b in zip(kept, k0))
    vm = m.get("view_mix", {})
    return [f"gs {m['grid_score']['mean_all']:.3f}   ret {ret:.2f}" + (f"   view_mix {vm['edge_frac']:.3f}" if vm else "")]


def _row_label(text, w, h):
    out = np.full((h, 36, 3), 245, np.uint8)
    lab = np.full((36, h, 3), 245, np.uint8)
    cv2.putText(lab, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    return np.rot90(lab).copy() if h > 0 else out


def compare_case(run_dir, cid, methods, device=None):
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    case_src = os.path.join(cfg["benchmark_dir"], "cases", cid)
    case = common.read_json(os.path.join(case_src, "case.json"))
    dataset = cfg.get("dataset_path") or common.DEFAULT_DATASET_PATH
    od = os.path.join(run_dir, "compare", cid)
    os.makedirs(od, exist_ok=True)
    labels = common.masks_to_labels([common.load_mask_png(os.path.join(case_src, f), case["mask_res"])
                                     for f in case["mask_files"]])
    heads = [common.load_hairstyle(h["npz_path"], cfg.get("max_strands"), seed=case["seed"] + k, root_uv_only=True)
             for k, h in enumerate(case["heads"])]
    region = common.hair_region([h["root_uv"] for h in heads], labels.shape[0])
    drawn = drawn_texture(labels, region)
    mask_2d(drawn, os.path.join(od, "mask_2d.png"))
    if not os.path.exists(os.path.join(od, "mask_scalp_views.png")):
        scalp_mask_views(drawn, dataset, os.path.join(od, "mask_scalp_views.png"))
    orig = original_renders(case, od, device)
    names = [f"{h['name']} ({h.get('length_class', '?')})" for h in case["heads"]]
    W = PANEL
    row1 = [_cell(["original base", names[0][:40]], orig[0]),
            _cell(["original donor", names[1][:40]], orig[1]),
            _cell(["drawn mask (2D)", f"{case['mask_generator']}; top = forehead"], os.path.join(od, "mask_2d.png")),
            _cell(["drawn mask on scalp", "front / top / left / back"], os.path.join(od, "mask_scalp_views.png")),
            _legend()]
    ref = common.read_json(os.path.join(run_dir, "cases", cid, "mask_only", "result.json"))
    row2, row3, row4 = [], [], []
    for mt in methods:
        md = os.path.join(run_dir, "cases", cid, mt)
        row2.append(_cell([mt] + _metrics(os.path.join(md, "result.json"), ref),
                          os.path.join(md, "render_optical", "front_multiview.png")))
        sv, m2 = os.path.join(od, f"kept_scalp_{mt}.png"), os.path.join(od, f"kept_2d_{mt}.png")
        if os.path.exists(os.path.join(md, "maps.npz")):
            d = np.load(os.path.join(md, "maps.npz"))
            tex = kept_texture([h["root_uv"] for h in heads], [d[f"keep_{k}"] for k in range(len(heads))])
            mask_2d(tex, m2, size=W // 2)
            scalp_mask_views(tex, dataset, sv)
        row3.append(_img(sv, W, W))
        row4.append(_img(m2, W, W // 2))
    ncol = max(len(row1), len(methods))
    blank = lambda h: np.full((h, W, 3), 255, np.uint8)  # noqa: E731

    def hrow(cells, h, label):
        cells = cells + [blank(h)] * (ncol - len(cells))
        sep = np.full((h, 6, 3), 255, np.uint8)
        out = cells[0]
        for c in cells[1:]:
            out = np.hstack([out, sep, c])
        return np.hstack([_row_label(label, 36, h), out])

    rows = [hrow(row1, row1[0].shape[0], "inputs"), hrow(row2, row2[0].shape[0], "fused hair (optical)"),
            hrow(row3, W, "kept-strand mask on scalp"), hrow(row4, W // 2, "kept mask 2D")]
    gap = lambda: np.full((14, rows[0].shape[1], 3), 255, np.uint8)  # noqa: E731
    line = np.full((2, rows[0].shape[1], 3), 200, np.uint8)
    title = _text_block([f"{cid}   {case['length_pair']}   mask: {case['mask_generator']}"], rows[0].shape[1], 40)
    img = np.vstack([title, rows[0], gap(), line, gap(), rows[1], gap(), rows[2], rows[3], gap()])
    out = os.path.join(run_dir, "compare", f"compare_{cid}.png")
    cv2.imwrite(out, img)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--cases", default=None, help="comma list (default: every case of the run)")
    ap.add_argument("--methods", default="mask_only,tsai_joint_fm,afm_a2_guard0.7,edit_region_lift,edit_guardfm_region_lift")
    ap.add_argument("--device", default=None, help="CYCLES_DEVICE for the original renders (CUDA if OptiX crashes)")
    a = ap.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    cases = a.cases.split(",") if a.cases else sorted(os.listdir(os.path.join(run_dir, "cases")))
    methods = [m for m in a.methods.split(",") if m]
    outs = []
    for cid in cases:
        print(cid)
        outs.append(compare_case(run_dir, cid, methods, a.device))
    html = ["<!doctype html><meta charset='utf-8'><title>Fusion comparison</title>",
            "<style>body{font-family:sans-serif;background:#fff;margin:16px}img{max-width:100%;display:block;"
            "margin-bottom:24px;border:1px solid #ddd}</style>",
            f"<h2>{os.path.basename(run_dir)}: optical renders</h2>",
            "<p>Per case: inputs (originals, drawn mask), then for each method the fused hair (optical render) and the "
            "mask of the strands it keeps (scalp views, 2D). Base = dark brown, donor = golden, red = bald.</p>"]
    html += [f"<img src='{os.path.basename(o)}' alt='{os.path.basename(o)}'>" for o in outs]
    with open(os.path.join(run_dir, "compare", "index.html"), "w") as f:
        f.write("\n".join(html))
    print("->", os.path.join(run_dir, "compare"))


if __name__ == "__main__":
    main()
