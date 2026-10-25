#!/usr/bin/env python3
"""Build one self-contained HTML report (all images embedded) that compares fusion results.

    python make_report.py <dir> [<dir> ...] [-o report.html]

Each <dir> is either one method's output (run_fusion.sh's <output_path>/kung or /tsai) or a
run_fusion.sh output_path holding kung/ and/or tsai/ - every method dir found becomes one
"variant". Every figure is shown as a table with ONE ROW PER VARIANT, so the same figure of
different runs lines up vertically; click an image to zoom, then <- / -> flips through the
variants on that same figure (Esc closes). Hover an image for its source file.

run_fusion.sh calls this at the end with its output_path (-> <output_path>/report.html).
Only needs numpy + Pillow (no torch / Blender), so it can be rerun on old outputs, e.g. to
compare an aligned and an unaligned run:
    python make_report.py outputs_a/tsai outputs_b/tsai -o compare.html
"""
import argparse
import base64
import datetime
import html
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_CONFIG = os.path.join(SCRIPT_DIR, "..", "configs", "config_scalp_texture_conditional.json")
FLUX_HEADS = ("base", "donor", "fused_before", "fused_after")


# ----------------------------------------------------------------------------- loading

def read_json(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def is_method_dir(d):
    return any(os.path.exists(os.path.join(d, x)) for x in
               ("blender_scene.blend", "mixed_output_strands.npz", "render", "analysis", "masks"))


def find_method_dirs(path):
    path = os.path.abspath(path)
    if is_method_dir(path):
        return [path]
    return [os.path.join(path, d) for d in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, d)) and is_method_dir(os.path.join(path, d))]


def newest_render_mtime(d):
    p = os.path.join(d, "render", "render_multiview.png")
    return os.path.getmtime(p) if os.path.exists(p) else -1


class Variant:
    def __init__(self, d):
        self.dir = d
        self.info = read_json(os.path.join(d, "run_info.json")) or {}
        self.method = self.info.get("method") or (
            "tsai" if any(os.path.exists(os.path.join(d, x)) for x in
                          ("analysis", "individual_strands", "individual_strands_aligned")) else "kung")
        self.stats = read_json(os.path.join(d, "analysis", "stats.json"))

        if self.method == "tsai":
            if "align_hairline" in self.info:
                self.aligned = bool(self.info["align_hairline"])
            else:   # older runs without run_info.json: the individual dir whose heads were rendered last
                cands = [c for c in ("individual_strands_aligned", "individual_strands")
                         if os.path.isdir(os.path.join(d, c))]
                self.aligned = bool(cands) and max(
                    cands, key=lambda c: max([newest_render_mtime(os.path.join(d, c, t))
                                              for t in os.listdir(os.path.join(d, c))] + [-1])
                ) == "individual_strands_aligned"
            self.ind_dir = os.path.join(d, "individual_strands_aligned" if self.aligned else "individual_strands")
        else:
            self.aligned = os.path.isdir(os.path.join(d, "debug_shifting_refined"))
            self.ind_dir = None
        self.align_dir = (os.path.join(d, "debug_shifting_refined") if self.method == "kung"
                          else os.path.join(self.ind_dir, "debug_shifting_refined"))
        self.alignment = read_json(os.path.join(self.ind_dir, "alignment.json")) if self.ind_dir else None

        tags = sorted(t for t in os.listdir(self.ind_dir)) if self.ind_dir and os.path.isdir(self.ind_dir) else []
        self.base_tag = next((t for t in tags if t.startswith("base_")), None)
        self.donor_tag = next((t for t in tags if t.startswith("0_")), None)

        run = os.path.basename(os.path.dirname(d))
        self.label = f"{run}/{os.path.basename(d)}"
        self.sublabel = self.method + (" + hairline align" if self.aligned else "")

    def p(self, *parts):
        """absolute path if it exists, else None"""
        if any(x is None for x in parts):
            return None
        path = os.path.join(self.dir, *parts)
        return path if os.path.exists(path) else None

    def head(self, which, *parts):
        tag = self.base_tag if which == "base" else self.donor_tag
        return self.p(os.path.relpath(self.ind_dir, self.dir), tag, *parts) if tag else None


def npz_array_shape(npz_path, key):
    """shape of one array in an .npz without loading it (positions is ~100 MB)"""
    try:
        with zipfile.ZipFile(npz_path) as z, z.open(key + ".npy") as f:
            version = np.lib.format.read_magic(f)
            return np.lib.format._read_array_header(f, version)[0]
    except Exception:
        return None


_SIGMA_DATA = None


def density_from_npz(npz_path, key="scalp_texture"):
    """[0, 1] density image from a saved raw scalp texture (for runs made before density.png existed)"""
    global _SIGMA_DATA
    if not npz_path:
        return None
    if _SIGMA_DATA is None:
        _SIGMA_DATA = (read_json(DIFFUSION_CONFIG) or {}).get("model", {}).get("sigma_data", 0.3)
    t = np.load(npz_path)[key]
    return np.clip(t[0, -1] * (0.5 / _SIGMA_DATA) + 0.5, 0, 1)


def gray(src):
    """path / array -> float32 HxW in [0, 1]"""
    if src is None:
        return None
    if isinstance(src, np.ndarray):
        return src.astype(np.float32)
    return np.asarray(Image.open(src).convert("L"), dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------- images

_cache = {}


def to_data_uri(src, max_w=1800):
    """path or PIL image -> data URI; small PNGs are embedded as-is, the rest re-encoded (renders are
    5400 px wide - downscaled to max_w, which still shows single strands when zoomed)."""
    key = (src if isinstance(src, str) else id(src), max_w)
    if key in _cache:
        return _cache[key]
    if isinstance(src, str) and src.lower().endswith(".png") and os.path.getsize(src) < 250_000 \
            and Image.open(src).width <= max_w:
        uri = "data:image/png;base64," + base64.b64encode(open(src, "rb").read()).decode()
    else:
        im = Image.open(src) if isinstance(src, str) else src
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        im = im.convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        if im.width * im.height <= 512 * 512:
            im.save(buf, "PNG")
            uri = "data:image/png;base64,"
        else:
            im.save(buf, "JPEG", quality=88)
            uri = "data:image/jpeg;base64,"
        uri += base64.b64encode(buf.getvalue()).decode()
    _cache[key] = uri
    return uri


def gray_to_image(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def overlap_image(a, b):
    """hairline overlap of two density maps: magenta = base only, green = donor only, white = both"""
    if a is None or b is None or a.shape != b.shape:
        return None
    return Image.fromarray((np.stack([a, b, a], -1) * 255).clip(0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------- html

class Cell:
    """one image in the comparison grid. src: file path, PIL image or None (= not available)"""

    def __init__(self, src, caption=None, pixelated=False, max_w=1800):
        self.src, self.caption, self.pixelated, self.max_w = src, caption, pixelated, max_w


def esc(x):
    return html.escape(str(x))


def comparison(sec_id, title, desc, columns, rows_fn, variants, wide=False):
    """one figure table: header row = column names, then one row per variant.
    rows_fn(variant) -> list of Cell (len(columns)); skipped entirely if no variant has any image."""
    rows = [(v, rows_fn(v)) for v in variants]
    if not any(c.src is not None for _, cells in rows for c in cells):
        return ""
    ncol = len(columns)
    style = f"grid-template-columns: var(--label-w) repeat({ncol}, minmax(0, 1fr));"
    out = [f'<section id="{sec_id}"><h2>{esc(title)}</h2>']
    if desc:
        out.append(f'<p class="desc">{desc}</p>')
    out.append(f'<div class="grid{" wide" if wide else ""}" style="{style}">')
    out.append('<div class="colhead"></div>' + "".join(f'<div class="colhead">{esc(c)}</div>' for c in columns))
    for v, cells in rows:
        vi = v.idx
        out.append(f'<div class="rowlabel v{vi}"><b>{esc(v.label)}</b><span>{esc(v.sublabel)}</span></div>')
        for ci, c in enumerate(cells):
            if c.src is None:
                out.append(f'<div class="cell v{vi} na">n/a</div>')
                continue
            uri = to_data_uri(c.src, c.max_w)
            path = os.path.relpath(c.src, os.getcwd()) if isinstance(c.src, str) else "(computed in report)"
            cap = f'<div class="cap">{esc(c.caption)}</div>' if c.caption else ""
            out.append(
                f'<div class="cell v{vi}"><img src="{uri}" title="{esc(path)}"'
                f' class="{"px" if c.pixelated else ""}" data-group="{sec_id}-{ci}"'
                f' data-label="{esc(v.label + " · " + v.sublabel + " — " + columns[ci])}">{cap}</div>')
    out.append("</div></section>")
    return "\n".join(out)


def fmt_num(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return f"{x:,}"


def arrow(a, b, digits=4):
    if a is None and b is None:
        return "—"
    return f"{fmt_num(a, digits)} → {fmt_num(b, digits)}"


def metrics_table(variants):
    def g(d, *keys):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def fused_counts(v):
        npz = v.p("mixed_output_strands.npz")
        if v.stats:
            return g(v.stats, "kept", "after", "base"), g(v.stats, "kept", "after", "donor")
        if npz:
            shape = npz_array_shape(npz, "positions")
            return (shape[0] if shape else None), None
        return None, None

    rows = []

    def row(name, fn, hint=""):
        rows.append((name, hint, [fn(v) for v in variants]))

    def ins(v):
        paths = v.info.get("input_file_paths") or []
        return "<br>".join(esc(os.path.basename(x)) for x in paths) or "—"

    def masks(v):
        paths = v.info.get("mask_file_paths") or []
        return "<br>".join(esc(os.path.relpath(x, v.info.get("cwd", os.getcwd()))) for x in paths) or "—"

    def seconds(v):
        s = v.info.get("elapsed_s")
        return f"{s // 60}m {s % 60:02d}s" if isinstance(s, int) else "—"

    row("method", lambda v: esc(v.method))
    row("hairline alignment", lambda v: "on" if v.aligned else "off")
    row("seed", lambda v: esc(v.info.get("seed", "—")))
    row("config", lambda v: esc(os.path.basename(v.info.get("input_config", "")) or "—"))
    row("inputs", ins)
    row("masks", masks)
    row("finished", lambda v: esc(v.info.get("finished", "—")))
    row("pipeline time (fusion + render)", seconds)

    def fused_total(v):
        a, b = fused_counts(v)
        if a is None:
            return "—"
        return f"{a + b:,} ({a:,} base + {b:,} donor)" if b is not None else f"{a:,}"
    row("fused strands", fused_total)
    row("strands per head (base / donor)",
        lambda v: f"{fmt_num(g(v.stats, 'n_strands', 'base'))} / {fmt_num(g(v.stats, 'n_strands', 'donor'))}"
        if v.stats else "—")

    def align_field(field):
        def f(v):
            if not v.alignment:
                return "—"
            h = v.alignment["heads"]
            return " / ".join(fmt_num(h.get(t, {}).get(field)) for t in (v.base_tag, v.donor_tag))
        return f
    row("alignment shift, px (base / donor)", align_field("dynamic_shift"), "rows each head's features moved up")
    row("hairline gap, texels (base / donor)", align_field("gap_texels"), "texels the union density adds")

    row("FM conflict cells", lambda v: arrow(*(g(v.stats, "objectives", k, 0) for k in ("before", "after"))),
        "drawn mask → refined; lower is better")
    row("FM crossing strand pairs", lambda v: arrow(*(g(v.stats, "objectives", k, 1) for k in ("before", "after"))),
        "lower is better")
    row("kept base strands", lambda v: arrow(g(v.stats, "kept", "before", "base"), g(v.stats, "kept", "after", "base")))
    row("kept donor strands",
        lambda v: arrow(g(v.stats, "kept", "before", "donor"), g(v.stats, "kept", "after", "donor")))
    row("mean hull score", lambda v: arrow(g(v.stats, "mean_hull_score", "before"),
                                           g(v.stats, "mean_hull_score", "after")),
        "base/donor mixing; lower is better")
    for h in FLUX_HEADS:
        row(f"mean |flux|, {h}", lambda v, h=h: fmt_num(g(v.stats, "heads", h, "mean_abs_flux")))
    for h in FLUX_HEADS:
        row(f"flux edge cells, {h}", lambda v, h=h: fmt_num(g(v.stats, "heads", h, "n_edge_cells")))

    rows = [r for r in rows if any(x not in ("—", "— / —", "— → —") for x in r[2])]
    out = ['<section id="metrics"><h2>Summary</h2>',
           '<p class="desc">One column per variant. Numbers come from <code>run_info.json</code>, '
           '<code>analysis/stats.json</code> and <code>alignment.json</code>; "a → b" = drawn mask → '
           'after joint FM refinement.</p><div class="tablewrap"><table><thead><tr><th></th>']
    out += [f'<th class="v{i}">{esc(v.label)}<span>{esc(v.sublabel)}</span></th>' for i, v in enumerate(variants)]
    out.append("</tr></thead><tbody>")
    for name, hint, vals in rows:
        h = f'<span class="hint">{esc(hint)}</span>' if hint else ""
        out.append(f"<tr><th>{esc(name)}{h}</th>" + "".join(f'<td class="v{i}">{x}</td>' for i, x in enumerate(vals))
                   + "</tr>")
    out.append("</tbody></table></div></section>")
    return "\n".join(out)


# ----------------------------------------------------------------------------- sections

def build_sections(variants):
    S = []
    S.append(metrics_table(variants))

    # inputs + masks
    n_in = max([len(v.info.get("input_file_paths") or []) for v in variants] + [2])
    n_mask = max([len(v.info.get("mask_file_paths") or []) for v in variants] + [1])

    def inputs(v):
        paths = v.info.get("input_file_paths") or []
        cells = []
        for i in range(n_in):
            src = paths[i] if i < len(paths) and os.path.exists(paths[i]) else None
            if src is None and v.ind_dir and i < 2:   # no run_info: the face crops the model saw
                src = v.head("base" if i == 0 else "donor", "rgb.png")
            cells.append(Cell(src, max_w=900))
        for k in range(1, n_mask + 1):
            cells.append(Cell(v.p("masks", f"mask_M{k}_raw.png") or v.p("masks", f"mask_M{k}.png"), pixelated=True))
        cells.append(Cell(v.p("masks", "mask_M1_refined.png"), pixelated=True))
        return cells
    S.append(comparison(
        "inputs", "Inputs and fusion masks",
        "Masks are in the scalp-texture layout (white = donor region). The refined mask is tsai's per-texel view "
        "of the per-strand joint-FM result.",
        [f"input {i + 1}" + (" (base)" if i == 0 else " (donor)" if i == 1 else "") for i in range(n_in)]
        + [f"mask M{k} (drawn)" for k in range(1, n_mask + 1)] + ["mask M1 (refined)"],
        inputs, variants))

    S.append(comparison(
        "render", "Fused hair", "White = base strands, black = donor strands (tsai); kung uses its color map.",
        ["render (5 views)"], lambda v: [Cell(v.p("render", "render_multiview.png"), max_w=2700)], variants, wide=True))

    S.append(comparison(
        "heads", "Each input hairstyle (tsai)",
        "The whole reconstructed head. Strands kept in the fused hair are in that head's color "
        "(base white, donor black); excluded strands are gray.",
        ["base", "donor"],
        lambda v: [Cell(v.head(w, "render", "render_multiview.png"), max_w=2700) for w in ("base", "donor")],
        [v for v in variants if v.method == "tsai"], wide=True))

    # hairline / density
    def densities(v):
        if v.method == "kung":
            a = v.p("intermediates", "density", "density_1.png")
            b = v.p("intermediates", "density", "density_2.png")
            union = v.p("debug_shifting_refined", "D_union_soft.png")
            ga, gb = gray(a), gray(b)
        else:
            a = v.head("base", "density.png")
            b = v.head("donor", "density.png")
            ga = gray(a) if a else density_from_npz(v.head("base", "scalp_texture.npz"))
            gb = gray(b) if b else density_from_npz(v.head("donor", "scalp_texture.npz"))
            a = a or (gray_to_image(ga) if ga is not None else None)
            b = b or (gray_to_image(gb) if gb is not None else None)
            union = None
            if v.aligned:
                union = v.head("base", "density_aligned.png")
                if union is None:
                    u = density_from_npz(v.head("base", "scalp_texture_aligned.npz"))
                    union = gray_to_image(u) if u is not None else None
        return [Cell(a, pixelated=True), Cell(b, pixelated=True),
                Cell(overlap_image(ga, gb), pixelated=True),
                Cell(union, pixelated=True),
                Cell(os.path.join(v.align_dir, "gap_mask_soft_1.png") if v.aligned and v.align_dir
                     and os.path.exists(os.path.join(v.align_dir, "gap_mask_soft_1.png")) else None, pixelated=True),
                Cell(os.path.join(v.align_dir, "gap_mask_soft_2.png") if v.aligned and v.align_dir
                     and os.path.exists(os.path.join(v.align_dir, "gap_mask_soft_2.png")) else None, pixelated=True)]
    S.append(comparison(
        "hairline", "Hairline: density before / after alignment",
        "Scalp-texture density of each head as diffusion produced it, and their overlap "
        '(<span class="mag">magenta</span> = base only, <span class="grn">green</span> = donor only, white = both): '
        "colored bands along the front are the hairline mismatch. With alignment both heads get the blurred union "
        "density; the gap masks show where each head was filled with its own shifted features. "
        "(kung's union image is its raw debug output, so its gray levels differ.)",
        ["base density", "donor density", "overlap", "aligned (union) density", "base gap mask", "donor gap mask"],
        densities, variants))

    S.append(comparison(
        "roots", "Strand roots and joint-FM masks (tsai)",
        "Strand roots in the scalp-texture layout (kept in the head's color, excluded gray), the grid mask, and the "
        "convex-hull mixing score, each before (drawn mask) and after joint FM.",
        ["strand roots", "grid mask", "hull score"],
        lambda v: [Cell(v.p("analysis", "figures", f)) for f in
                   ("strand_roots.png", "fused_grid_mask.png", "fused_hull_score.png")],
        [v for v in variants if v.method == "tsai"]))

    for fig, title, desc in (
            ("flux_with_flow.png", "Flux + flow direction (tsai)",
             "3D |flux| (divergence) of the hair flow per scalp-grid cell, with strand control-point directions."),
            ("flux_score.png", "Flux score (tsai)", "3D |flux| per scalp-grid cell."),
            ("flux_edges.png", "Flux edges (tsai)", "Sobel / Canny edges of the |flux| map.")):
        S.append(comparison(
            "flux-" + fig.split(".")[0], title, desc + " Scalp-grid layout (not the mask's layout).",
            list(FLUX_HEADS),
            lambda v, fig=fig: [Cell(v.p("analysis", "figures", h, fig)) for h in FLUX_HEADS],
            [v for v in variants if v.method == "tsai"]))

    S.append(comparison(
        "textures", "Scalp textures (kung)", "PCA view of the latent scalp textures and kung's blended result.",
        ["texture 1", "texture 2", "texture 1 shifted", "texture 2 shifted", "blended"],
        lambda v: [Cell(v.p(f), pixelated=True) for f in
                   ("scalp_texture_1.png", "scalp_texture_2.png", "debug_shifting_refined/refined_shift_texture_1.png",
                    "debug_shifting_refined/refined_shift_texture_2.png", "scalp_texture_FINAL_MIXED.png")],
        [v for v in variants if v.method == "kung"]))
    return [s for s in S if s]


CSS = """
:root { --bg:#f6f7f9; --panel:#fff; --text:#1d2330; --muted:#647084; --line:#dde2ea; --accent:#2f6fd6;
        --na:#eef1f5; --label-w:170px; --fig-h:520px; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171c; --panel:#1c2027; --text:#e4e8ef; --muted:#97a1b3;
        --line:#2e3440; --accent:#7aa7ff; --na:#232831; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui, -apple-system, "Segoe UI",
       "Noto Sans TC", sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; }
header nav { display:flex; gap:4px 12px; flex-wrap:wrap; font-size:13px; }
header nav a { color:var(--accent); text-decoration:none; }
header .toggles { margin-left:auto; display:flex; gap:10px; font-size:13px; flex-wrap:wrap; }
main { padding:0 16px 48px; max-width:2200px; margin:0 auto; }
section { margin-top:28px; }
h2 { font-size:17px; margin:0 0 4px; }
.desc { color:var(--muted); margin:0 0 10px; max-width:1100px; }
.mag { color:#d03ad0; font-weight:600; } .grn { color:#1f9d3a; font-weight:600; }
.grid { display:grid; gap:6px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px; }
.grid.wide { --label-w:150px; }
.colhead { font-size:12px; font-weight:600; color:var(--muted); text-align:center; align-self:end; }
.rowlabel { font-size:12px; display:flex; flex-direction:column; justify-content:center; padding-right:6px;
            word-break:break-all; border-top:1px solid var(--line); }
.rowlabel span { color:var(--muted); }
.cell { border-top:1px solid var(--line); padding-top:6px; min-width:0; text-align:center; }
.cell img { max-width:100%; max-height:var(--fig-h); width:auto; height:auto; display:block; margin:0 auto;
            cursor:zoom-in; border-radius:3px; outline:1px solid var(--line); }
.grid.wide .cell img { width:100%; max-height:none; }
.cell img.px { image-rendering:pixelated; width:100%; max-width:320px; }
.cell.na { background:var(--na); color:var(--muted); display:flex; align-items:center; justify-content:center;
           min-height:60px; border-radius:4px; font-size:12px; }
.cap { font-size:11px; color:var(--muted); }
.tablewrap { overflow-x:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:5px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
thead th { position:sticky; top:0; background:var(--panel); }
thead th span, th .hint { display:block; font-weight:400; color:var(--muted); font-size:12px; }
td { font-variant-numeric:tabular-nums; }
.hidden { display:none !important; }
#lb { position:fixed; inset:0; background:rgba(0,0,0,.92); z-index:10; display:none; flex-direction:column; }
#lb.on { display:flex; }
#lb .bar { color:#eee; padding:8px 16px; font-size:13px; display:flex; gap:16px; }
#lb .bar b { flex:1; }
#lb .view { flex:1; overflow:auto; display:flex; align-items:flex-start; justify-content:center; }
#lb img { max-width:100%; cursor:zoom-in; image-rendering:auto; }
#lb img.px { image-rendering:pixelated; min-width:min(90vh, 100%); }
#lb img.full { max-width:none; cursor:zoom-out; }
@media (max-width:700px) { :root { --label-w:90px; } main { padding:0 8px 32px; } }
"""

JS = """
const lb = document.getElementById('lb'), lbImg = lb.querySelector('img'), lbTitle = lb.querySelector('b'),
      lbPos = lb.querySelector('.pos');
let group = [], idx = 0;
function show() {
  const im = group[idx];
  lbImg.src = im.src; lbImg.className = im.className; lbTitle.textContent = im.dataset.label;
  lbPos.textContent = group.length > 1 ? `${idx + 1} / ${group.length}   ← → switch variant` : '';
}
document.querySelectorAll('.cell img').forEach(im => im.addEventListener('click', () => {
  group = [...document.querySelectorAll(`img[data-group="${im.dataset.group}"]`)]
          .filter(x => !x.closest('.hidden'));
  idx = group.indexOf(im); show(); lb.classList.add('on');
}));
lbImg.addEventListener('click', e => { e.stopPropagation(); lbImg.classList.toggle('full'); });
lb.addEventListener('click', () => lb.classList.remove('on'));
document.addEventListener('keydown', e => {
  if (!lb.classList.contains('on')) return;
  if (e.key === 'Escape') lb.classList.remove('on');
  else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { idx = (idx + 1) % group.length; show(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { idx = (idx - 1 + group.length) % group.length; show(); e.preventDefault(); }
});
document.querySelectorAll('.toggles input').forEach(cb => cb.addEventListener('change', () =>
  document.querySelectorAll('.' + cb.value).forEach(el => el.classList.toggle('hidden', !cb.checked))));
"""


def build_report(paths, out):
    dirs = [d for p in paths for d in find_method_dirs(p)]
    if not dirs:
        raise SystemExit(f"no fusion output found in {paths}")
    variants = [Variant(d) for d in dirs]
    for i, v in enumerate(variants):
        v.idx = i   # css class v<idx> on every row / column of this variant (for the show/hide toggles)
    sections = build_sections(variants)
    nav_titles = {"metrics": "Summary", "inputs": "Inputs", "render": "Fused hair", "heads": "Each head",
                  "hairline": "Hairline", "roots": "Roots / FM", "flux-flux_with_flow": "Flux + flow",
                  "flux-flux_score": "Flux score", "flux-flux_edges": "Flux edges", "textures": "Textures"}
    nav = "".join(f'<a href="#{k}">{t}</a>' for k, t in nav_titles.items() if f'id="{k}"' in "".join(sections))
    toggles = "".join(f'<label><input type="checkbox" value="v{i}" checked> {esc(v.label)}</label>'
                      for i, v in enumerate(variants)) if len(variants) > 1 else ""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fusion Report</title><style>{CSS}</style></head>
<body><header><h1>Fusion report <span style="font-weight:400;color:var(--muted)">· {len(variants)} variant(s) · {now}</span></h1>
<nav>{nav}</nav><div class="toggles">{toggles}</div></header>
<main>{"".join(sections)}</main>
<div id="lb"><div class="bar"><b></b><span class="pos"></span><span>click image: 1:1 · Esc: close</span></div>
<div class="view"><img alt=""></div></div>
<script>{JS}</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"report: {out} ({len(variants)} variants: {', '.join(v.label + ' [' + v.sublabel + ']' for v in variants)}, "
          f"{os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="method output dirs, or run_fusion.sh output_path dirs")
    ap.add_argument("-o", "--out", default=None, help="default: <first path>/report.html")
    args = ap.parse_args()
    build_report(args.paths, args.out or os.path.join(args.paths[0], "report.html"))
