"""HTML report of a benchmark run (written by summarize.py into the run folder).

    report.html           index: hair length classes, overview table (grid score by length pair, links to
                          the pair pages), and the "all cases" group
    report_<pair>.html    one page per hair-length pair (e.g. report_short-long.html)

Every group shows only averages: per-method mean ± std of the KEY_METRICS and the paired comparison against
the reference method. It also shows its lowest and highest grid-score cases (picked by
summary["select_method"]), each with the drawn mask, every method's multi-view render, and
img2hair_tsai_1.debug_field_analysis' figures (analysis.py): the |flux| map and the flow vector field of the
two input hairstyles and of the fused hair before (reference) / after (method), plus the grid-score map.

Images are linked by relative path, not embedded: keep the report pages inside the run folder (and the
benchmark data folder where it is), or copy them together.
"""
import html
import os

import common
from analysis import FIGURE_HEADS, analysis_dir

CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --line:#e3e2dd;
        --good:#1f8a4c; --bad:#c2410c; --chip:#efeee9; --chip-on:#1d1d1b; --chip-on-fg:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29c; --card:#1f1f1d; --line:#34332f;
          --good:#4cc782; --bad:#f08a50; --chip:#2a2a27; --chip-on:#ecebe6; --chip-on-fg:#161615; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, -apple-system, "Noto Sans TC", sans-serif; }
main { max-width:1400px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 10px; }
h3 { font-size:15px; margin:18px 0 8px; } h4 { font-size:13.5px; margin:14px 0 6px; }
.muted { color:var(--muted); } code { font-size:12.5px; }
nav { margin:14px 0 0; display:flex; flex-wrap:wrap; gap:6px; }
nav a { background:var(--chip); color:var(--fg); text-decoration:none; padding:3px 10px; border-radius:99px;
        font-size:13px; }
nav a.on { background:var(--chip-on); color:var(--chip-on-fg); }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:5px 10px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { font-weight:600; color:var(--muted); font-size:12.5px; }
.good { color:var(--good); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:0 0 16px; min-width:0; }
.card .head { display:grid; grid-template-columns:160px 1fr; gap:14px; align-items:start; margin-top:8px; }
.label { font-weight:600; font-size:13px; margin:12px 0 4px; }
img { width:100%; height:auto; display:block; border-radius:6px; background:#fff; cursor:zoom-in; }
.figs { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px 14px; }
.figs .wide { grid-column:1 / -1; }
@media (max-width: 800px) { .figs { grid-template-columns:1fr; } .card .head { grid-template-columns:1fr; } }
details { margin-top:12px; } summary { cursor:pointer; font-weight:600; }
.note { color:var(--muted); font-size:12.5px; }
.kv td, .kv th { padding:2px 8px; border:0; }
#lightbox { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center;
            justify-content:center; padding:16px; z-index:10; cursor:zoom-out; }
#lightbox img { max-width:100%; max-height:100%; width:auto; }
"""

JS = """
const lb = document.getElementById('lightbox'), lbi = lb.querySelector('img');
document.addEventListener('click', e => {
  if (e.target.tagName === 'IMG' && !e.target.closest('#lightbox')) { lbi.src = e.target.src; lb.style.display = 'flex'; }
  else if (e.target.closest('#lightbox')) lb.style.display = 'none';
});
"""

GS = "grid_score.mean_all"
TYPICAL = {"short": "buzz / crew cut / pixie, lies on the scalp", "mid": "covers the ears, down to about the chin",
           "long": "past the chin, shoulder length and longer"}
e = html.escape


def page_name(group):
    return "report.html" if group == "all" else f"report_{group.replace('+', '-')}.html"


def _delta_cls(delta, lower):
    if lower is None or delta is None or delta == 0:
        return ""
    return "good" if (delta < 0) == lower else "bad"


def _rel(d):
    return f"{d['mean_rel_delta']:+.1%}" if d.get("mean_rel_delta") is not None else "-"


def _page(run_dir, group, title, body, groups, cfg):
    nav = "".join(f"<a class='{'on' if g == group else ''}' href='{page_name(g)}'>"
                  f"{'all (index)' if g == 'all' else e(g)} · {groups[g]['n_cases']}</a>" for g in groups)
    html_ = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{e(title)}</title>"
             f"<style>{CSS}</style></head><body><main><h1>{e(title)}</h1>"
             f"<div class='muted'>run <code>{e(run_dir)}</code><br>benchmark <code>{e(str(cfg.get('benchmark_dir')))}"
             f"</code> · max_strands {e(str(cfg.get('max_strands')))} · analysis grid {e(str(cfg.get('grid_n')))} · "
             f"generated {common.now_str()}</div><nav>{nav}</nav>{body}"
             "<p class='note' style='margin-top:32px'>Renders: base strands white, donor strands black. Mask: blue = "
             "base, orange = donor (scalp-texture layout). Analysis figures (img2hair_tsai_1.debug_field_analysis): "
             "base blue, donor orange, excluded strands gray; maps in the scalp-grid layout.</p>"
             f"</main><div id='lightbox'><img alt=''></div><script>{JS}</script></body></html>")
    path = os.path.join(run_dir, page_name(group))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_)
    return path


def _img(run_dir, path, alt, missing):
    if os.path.exists(path):
        return f"<img loading='lazy' src='{e(os.path.relpath(path, run_dir))}' alt='{e(alt)}'>"
    return f"<div class='note'>{e(missing)}</div>"


def _extreme_card(run_dir, which, x, methods, reference, sel, bench_dir):
    cid = x["case"]
    scores = "".join(f"<tr><th>{e(m)}</th><td>{v:.4f}</td></tr>" for m, v in x["grid_score"].items() if v is not None)
    lengths = " + ".join(f"{c} ({float(l) * 100:.1f} cm)" for c, l in
                         zip(x["length_classes"].split("+"), x["median_lengths"].split("+")))
    warn = (f"<div class='bad' style='margin-top:4px'>Degenerate: {e(sel)} kept only "
            f"{(x.get('min_head_retention') or 0):.0%} of one head's mask-assigned strands (that head practically "
            "vanished; its grid score is trivially ~0). Shown because this group has no other result.</div>"
            if x.get("degenerate") else "")
    out = [f"<div class='card'><div><b>{which}</b> {e(GS)} · {e(cid)}</div>{warn}"
           f"<div class='muted' style='font-size:12.5px'>{e(x['heads'].replace('+', ' + '))} · {e(lengths)} · "
           f"mask {e(x['mask_generator'])}</div><div class='head'>"
           + _img(run_dir, os.path.join(bench_dir or "", "cases", cid, "labels.png"), "mask", "no mask preview")
           + f"<table class='kv'><tr><th colspan='2' style='text-align:left'>{e(GS)}</th></tr>{scores}</table></div>"]
    for m in methods:
        out.append(f"<div class='label'>render · {e(m)}</div>")
        out.append(_img(run_dir, os.path.join(run_dir, "cases", cid, m, "render", "render_multiview.png"),
                        f"{m} render", "not rendered (summarize.py without --no_render)"))

    fig = os.path.join(analysis_dir(run_dir, cid), "figures")
    if not os.path.isdir(fig):
        out.append("<div class='note' style='margin-top:10px'>no flux / vector field analysis "
                   "(summarize.py without --no_analysis)</div></div>")
        return "".join(out)
    head_title = {"base": "base (whole input hairstyle)", "donor": "donor (whole input hairstyle)",
                  "fused_before": f"fused, before FM = {reference}", "fused_after": f"fused, after FM = {sel}"}
    out.append("<details open><summary>Flux map &amp; vector field</summary>"
               "<p class='note'>Left: 3D |flux| map (divergence of the hair flow field; fused heads: each source's "
               "kept strands alone + the fused map colored by source). Right: |flux| in gray with the per control "
               "point flow vectors, colored by source.</p>")
    for h in FIGURE_HEADS:
        out.append(f"<h4>{e(head_title[h])}</h4><div class='figs'>")
        flux = os.path.join(fig, h, "flux_score.png")
        # the fused flux_score has three panels: give it the full row, the flow field below it
        wide = h.startswith("fused")
        out.append(f"<div class='{'wide' if wide else ''}'>" + _img(run_dir, flux, f"{h} flux", "missing") + "</div>")
        out.append("<div>" + _img(run_dir, os.path.join(fig, h, "flux_with_flow.png"), f"{h} vector field",
                                  "missing") + "</div></div>")
    out.append("<h4>grid score map (convex-hull overlap of base vs donor), before / after FM</h4>"
               + _img(run_dir, os.path.join(fig, "fused_hull_score.png"), "grid score map", "missing")
               + "<h4>grid mask + flow, before / after FM</h4>"
               + _img(run_dir, os.path.join(fig, "fused_grid_mask.png"), "grid mask", "missing")
               + "</details></div>")
    return "".join(out)


def _group_body(run_dir, name, g, summary, key_metrics, bench_dir):
    methods, reference, sel = summary["methods"], summary["reference"], summary["select_method"]
    lower_of = dict(key_metrics)
    out = [f"<h2>{'All cases' if name == 'all' else e(name)} <span class='muted'>({g['n_cases']} cases)</span></h2>"]
    key_cols = [c for c, _ in key_metrics if any(c in g["methods"][m]["stats"] for m in methods)]
    out.append("<h3>Mean ± std</h3><p class='note'>degenerate = results keeping less than "
               f"{summary.get('degenerate_retention', 0.1):.0%} of some head's mask-assigned strands (that head practically "
               "vanished, so crossings and grid score are trivially ~0). They are included in these averages. failed = no result, incl. <code>skipped_oom</code> (killed by the run's memory guard; hover for the cases).</p>"
               "<div class='tablewrap'><table><tr><th>method</th><th>ok</th><th>failed</th><th>degenerate</th>"
               + "".join(f"<th>{e(c)}</th>" for c in key_cols) + "</tr>")
    for m in methods:
        s = g["methods"][m]
        cells = "".join(f"<td>{s['stats'][c]['mean']:.4g} ± {s['stats'][c]['std']:.2g}</td>"
                        if c in s["stats"] else "<td>-</td>" for c in key_cols)
        dg = s.get("degenerate", [])
        dg_cell = f"<td class='bad' title='{e(' '.join(dg))}'>{len(dg)}</td>" if dg else "<td>0</td>"
        bad = s.get("not_ok", {})
        fail_cell = (f"<td class='bad' title='{e(' '.join(f'{c} ({st})' for c, st in sorted(bad.items())))}'>"
                     f"{s['failed']}</td>") if s["failed"] else "<td>0</td>"
        out.append(f"<tr><td>{e(m)}</td><td>{s['ok']}</td>{fail_cell}{dg_cell}{cells}</tr>")
    out.append("</table></div>")

    if g["vs_reference"]:
        out.append(f"<h3>Paired vs <code>{e(reference)}</code></h3><p class='note'>Same cases; win rate = share of "
                   "cases where the method is better than the reference (green = better on average).</p>"
                   "<div class='tablewrap'><table><tr><th>method</th><th>metric</th><th>n</th><th>mean delta</th>"
                   "<th>mean rel. delta</th><th>win rate</th></tr>")
        for m, cmp in g["vs_reference"].items():
            for c, d in cmp.items():
                cls = _delta_cls(d["mean_delta"], lower_of.get(c))
                wr = f"{d['win_rate']:.0%}" if "win_rate" in d else "-"
                out.append(f"<tr><td>{e(m)}</td><td>{e(c)}</td><td>{d['n']}</td>"
                           f"<td class='{cls}'>{d['mean_delta']:.4g}</td><td class='{cls}'>{_rel(d)}</td>"
                           f"<td>{wr}</td></tr>")
        out.append("</table></div>")

    excluded = g["methods"].get(sel, {}).get("degenerate", [])
    excl_note = (f"<p class='note'>Degenerate results excluded from the pick: {e(', '.join(excluded))}.</p>"
                 if excluded else "")
    if g["extremes"]:
        same = len({x["case"] for x in g["extremes"].values()}) == 1
        if any(x.get("degenerate") for x in g["extremes"].values()):
            excl_note = ""
        out.append(f"<h3>Lowest / highest {e(GS)} of <code>{e(sel)}</code></h3>{excl_note}"
                   + ("<p class='note'>Only one candidate case in this group.</p>" if same else ""))
        for which, x in g["extremes"].items():
            if not (same and which == "highest"):
                out.append(_extreme_card(run_dir, which, x, methods, reference, sel, bench_dir))
    return "".join(out)


def make_report(run_dir, summary, key_metrics):
    cfg_path = os.path.join(run_dir, "run_config.json")
    cfg = common.read_json(cfg_path) if os.path.exists(cfg_path) else {}
    bench_dir = cfg.get("benchmark_dir")
    methods, groups, reference = summary["methods"], summary["groups"], summary["reference"]
    others = [m for m in methods if m != reference]

    # ---- pair pages
    for name, g in groups.items():
        if name != "all":
            _page(run_dir, name, f"Fusion benchmark · {name}",
                  _group_body(run_dir, name, g, summary, key_metrics, bench_dir), groups, cfg)

    # ---- index: length classes, overview, all cases
    body = ["<h2>Hair length classes</h2><p class='note'>Median strand length of each hairstyle (guide strands). "
            "A case's pair is the unordered combination of its heads' classes.</p><div class='tablewrap'><table>"
            "<tr><th>class</th><th>median strand length</th><th>typical</th></tr>"]
    for d in summary["length_standard"]:
        hi = "∞" if d["max_m"] is None else f"{d['max_m'] * 100:g} cm"
        body.append(f"<tr><td>{e(d['name'])}</td><td>{d['min_m'] * 100:g} cm – {hi}</td>"
                    f"<td>{e(TYPICAL.get(d['name'], ''))}</td></tr>")
    body.append("</table></div>")

    body.append(f"<h2>Overview: {e(GS)} by length pair</h2><p class='note'>Mean over the group's cases. "
                f"Δ = method − <code>{e(reference)}</code> on the same cases (green = better); win = share of cases "
                "where the method is better. Click a pair for its page.</p><div class='tablewrap'><table><tr>"
                "<th>group</th><th>cases</th>" + "".join(f"<th>{e(m)}</th>" for m in methods)
                + "".join(f"<th>Δ {e(m)}</th><th>win</th><th>Δ crossing pairs</th><th>Δ bald frac</th>"
                          f"<th>degenerate</th>" for m in others) + "</tr>")
    for name, g in groups.items():
        cells = []
        for m in methods:
            s = g["methods"][m]["stats"].get(GS)
            cells.append(f"<td>{s['mean']:.4f}</td>" if s else "<td>-</td>")
        for m in others:
            cmp = g["vs_reference"].get(m, {})
            d = cmp.get(GS)
            cells.append(f"<td class='{_delta_cls(d['mean_delta'], True)}'>{d['mean_delta']:+.4f} ({_rel(d)})</td>"
                         f"<td>{d['win_rate']:.0%}</td>" if d else "<td>-</td><td>-</td>")
            for c in ("crossings.pairs", "coverage.bald_frac"):
                d = cmp.get(c)
                cells.append(f"<td class='{_delta_cls(d['mean_delta'], True)}'>{_rel(d)}</td>" if d else "<td>-</td>")
            dg = g["methods"][m].get("degenerate", [])
            cells.append(f"<td class='bad' title='{e(' '.join(dg))}'>{len(dg)}</td>" if dg else "<td>0</td>")
        label = "<b>all</b>" if name == "all" else f"<a href='{page_name(name)}'>{e(name)}</a>"
        body.append(f"<tr><td>{label}</td><td>{g['n_cases']}</td>{''.join(cells)}</tr>")
    body.append("</table></div>")
    body.append(_group_body(run_dir, "all", groups["all"], summary, key_metrics, bench_dir))
    return _page(run_dir, "all", "Fusion benchmark report", "".join(body), groups, cfg)
