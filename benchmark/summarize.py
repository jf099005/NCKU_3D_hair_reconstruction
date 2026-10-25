#!/usr/bin/env python3
"""Aggregate a benchmark run: python benchmark/summarize.py <run_dir> [--reference mask_only]
                                  [--select_method tsai_joint_fm] [--no_render]

Cases are grouped by their hair-length pair (common.LENGTH_CLASSES: short < 5 cm <= mid < 15 cm <= long,
by median strand length; the pair is the unordered combination of the heads' classes, e.g. "short+long").
Only averages are reported, for the whole benchmark ("all") and for each pair:

    summary.csv   one row per (case, method): length classes + every scalar metric ("<metric>.<key>[.<i>]")
    summary.json  per group: n cases; per method n ok / failed and mean / std / median of every numeric column;
                  per method vs the reference method, paired on the same cases: mean delta and win rate of
                  KEY_METRICS; the cases with the lowest and the highest grid score of --select_method among
                  the non-degenerate results (DEGENERATE_RETENTION); the degenerate cases per method
    summary.md    the KEY_METRICS tables of summary.json
    report.html   index: length classes, overview by pair, all cases; report_<pair>.html: one page per pair.
                  Each group shows its lowest / highest grid-score cases with renders, |flux| maps and flow
                  vector fields (report.py)

The lowest / highest cases of every group are rendered (all methods, render_results.render_case) unless
--no_render, and get img2hair_tsai_1's flux / vector-field analysis (analysis.analyze_case, reference vs
select_method) unless --no_analysis; existing renders / analyses are reused.
"""
import argparse
import csv
import glob
import os

import numpy as np

import common
import report

# (column, lower is better) - the headline numbers of summary.md / report.html
KEY_METRICS = [
    ("grid_score.mean_all", True),
    ("grid_score.mixed_mean", True),
    ("grid_score.frac_above", True),
    ("grid_score.per_black_area", True),
    ("grid_score.per_white_area", True),
    ("crossings.pairs", True),
    ("crossings.conflict_cells", True),
    ("coverage.bald_frac", True),
    ("flux.abs_mean", None),
    ("strands.donor_share", None),
    ("strands.changed_frac", None),
    ("min_head_retention", None),
    ("overlap_matrix.nnz", None),
    ("overlap_matrix.density", None),
    ("method_seconds", True),
]
SELECT_METRIC = "grid_score.mean_all"
# a result keeping less than this fraction of some head's mask-assigned strands (min_head_retention, relative
# to the reference = the drawn mask) is degenerate: that head practically vanished from the fused hair, so its
# crossings / grid score are trivially ~0. Degenerate results stay in the averages (counted separately) but are
# never picked as a group's lowest / highest case.
DEGENERATE_RETENTION = 0.1


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, (dict, list, tuple)):
                    continue
                out[f"{key}.{i}"] = x
        else:
            out[key] = v
    return out


def case_lengths(bench_dir, cid, dataset_path, cache):
    """(per-head median lengths, per-head classes, pair) of a case; benchmarks built before the length
    classes existed are classified here from the dataset"""
    if cid not in cache:
        case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
        lengths = [h.get("median_length") if h.get("median_length") is not None
                   else common.hairstyle_length(dataset_path, h["name"]) for h in case["heads"]]
        classes = [common.length_class(x) for x in lengths]
        cache[cid] = (lengths, classes, common.length_pair(classes))
    return cache[cid]


def load_rows(run_dir):
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench_dir = cfg["benchmark_dir"]
    dataset_path = common.read_json(os.path.join(bench_dir, "benchmark.json"))["dataset_path"]
    rows, cache = [], {}
    for path in sorted(glob.glob(os.path.join(run_dir, "cases", "*", "*", "result.json"))):
        r = common.read_json(path)
        lengths, classes, pair = case_lengths(bench_dir, r["case"], dataset_path, cache)
        row = dict(case=r["case"], method=r["method"], status=r.get("status"), m=r.get("m"),
                   mask_generator=r.get("mask_generator"), heads="+".join(r.get("heads", [])),
                   length_pair=pair, length_classes="+".join(classes),
                   median_lengths="+".join(f"{x:.3f}" for x in lengths),
                   method_seconds=r.get("method_seconds"))
        row.update(_flatten(r.get("metrics", {})))
        obj = r.get("method_info", {}).get("objectives")
        if obj:
            row.update(_flatten(obj, "fm_objectives."))
        ov = r.get("method_info", {}).get("overlap_matrix")
        if ov:
            row.update(_flatten(ov, "overlap_matrix."))
        rows.append(row)
    return rows


def add_retention(rows, reference):
    """min_head_retention = min over heads of kept / kept by the reference (the drawn mask), and the
    degenerate flag (< DEGENERATE_RETENTION); None where the reference result is missing"""
    ref = {r["case"]: r for r in rows if r["method"] == reference and r["status"] == "ok"}
    for r in rows:
        b = ref.get(r["case"])
        if r["status"] != "ok" or b is None:
            r["min_head_retention"], r["degenerate"] = None, None
            continue
        i, ret = 0, []
        while f"strands.kept.{i}" in r and f"strands.kept.{i}" in b:
            ret.append(r[f"strands.kept.{i}"] / b[f"strands.kept.{i}"] if b[f"strands.kept.{i}"] else 1.0)
            i += 1
        r["min_head_retention"] = round(min(ret), 4) if ret else None
        r["degenerate"] = bool(ret) and min(ret) < DEGENERATE_RETENTION


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def group_summary(rows, cols, methods, reference, select_method):
    """averages of one group of cases (all methods), paired comparison, lowest / highest select_method case"""
    ok = {mt: [r for r in rows if r["method"] == mt and r["status"] == "ok"] for mt in methods}
    g = dict(n_cases=len({r["case"] for r in rows}), methods={}, vs_reference={}, extremes={})
    for mt in methods:
        stats = {}
        for c in cols:
            v = np.array([r[c] for r in ok[mt] if _num(r.get(c))], dtype=np.float64)
            if v.size:
                stats[c] = dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)), n=int(v.size))
        not_ok = {r["case"]: r["status"] for r in rows if r["method"] == mt and r["status"] != "ok"}
        g["methods"][mt] = dict(ok=len(ok[mt]), failed=len(not_ok), not_ok=not_ok,
                                degenerate=sorted(r["case"] for r in ok[mt] if r.get("degenerate")), stats=stats)

    ref = {r["case"]: r for r in ok.get(reference, [])}
    for mt in methods:
        if mt == reference or not ref:
            continue
        cmp = {}
        for c, lower in KEY_METRICS:
            pairs = [(r[c], ref[r["case"]][c]) for r in ok[mt]
                     if r["case"] in ref and _num(r.get(c)) and _num(ref[r["case"]].get(c))]
            if not pairs:
                continue
            a, b = np.array(pairs, dtype=np.float64).T
            nz = np.abs(b) > 1e-12          # relative change only where the reference is nonzero
            e = dict(n=len(pairs), mean_delta=float((a - b).mean()),
                     mean_rel_delta=float(((a[nz] - b[nz]) / np.abs(b[nz])).mean()) if nz.any() else None)
            if lower is not None:
                e["win_rate"] = float(((a < b) if lower else (a > b)).mean())
            cmp[c] = e
        g["vs_reference"][mt] = cmp

    scored = [r for r in ok.get(select_method, []) if _num(r.get(SELECT_METRIC))]
    # degenerate results only when the group has nothing else (they are then marked as such)
    cand = [r for r in scored if not r.get("degenerate")] or scored
    if cand:
        for name, r in (("lowest", min(cand, key=lambda r: r[SELECT_METRIC])),
                        ("highest", max(cand, key=lambda r: r[SELECT_METRIC]))):
            g["extremes"][name] = dict(
                case=r["case"], heads=r["heads"], length_classes=r["length_classes"],
                median_lengths=r["median_lengths"], mask_generator=r["mask_generator"],
                degenerate=bool(r.get("degenerate")), min_head_retention=r.get("min_head_retention"),
                grid_score={mt: next((x.get(SELECT_METRIC) for x in ok[mt] if x["case"] == r["case"]), None)
                            for mt in methods})
    return g


def _md_tables(g, methods, key_cols, reference):
    lines = ["| method | ok | failed | degenerate | " + " | ".join(key_cols) + " |",
             "|---|---|---|---|" + "---|" * len(key_cols)]
    for mt in methods:
        s = g["methods"][mt]
        cells = [f"{s['stats'][c]['mean']:.4g} ± {s['stats'][c]['std']:.2g}" if c in s["stats"] else "-"
                 for c in key_cols]
        dg = f"{len(s['degenerate'])} {' '.join(s['degenerate'])}".strip()
        lines.append(f"| {mt} | {s['ok']} | {s['failed']} | {dg} | " + " | ".join(cells) + " |")
    if g["vs_reference"]:
        lines += ["", f"Paired vs `{reference}` (same cases; win = better than the reference):", "",
                  "| method | metric | n | mean delta | mean rel. delta | win rate |", "|---|---|---|---|---|---|"]
        for mt, cmp in g["vs_reference"].items():
            for c, e in cmp.items():
                wr = f"{e['win_rate']:.0%}" if "win_rate" in e else "-"
                rel = f"{e['mean_rel_delta']:+.1%}" if e["mean_rel_delta"] is not None else "-"
                lines.append(f"| {mt} | {c} | {e['n']} | {e['mean_delta']:.4g} | {rel} | {wr} |")
    for name, e in g["extremes"].items():
        scores = ", ".join(f"{m} {v:.4f}" for m, v in e["grid_score"].items() if v is not None)
        lines.append(f"\n{name} {SELECT_METRIC}: {e['case']} ({e['heads']}, {e['length_classes']}, "
                     f"{e['mask_generator']}) {scores}" + (" [DEGENERATE: only such results in this group]"
                                                           if e.get("degenerate") else ""))
    return lines


def _class_str(d):
    hi = "inf" if d["max_m"] is None else f"{d['max_m'] * 100:g}"
    return f"{d['name']} [{d['min_m'] * 100:g}, {hi}) cm"


def summarize(run_dir, reference="mask_only", select_method="tsai_joint_fm", render=True, analyze=True):
    rows = load_rows(run_dir)
    if not rows:
        print(f"no results in {run_dir}")
        return None
    add_retention(rows, reference)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    methods = list(dict.fromkeys(r["method"] for r in rows))
    if select_method not in methods:
        select_method = next((m for m in methods if m != reference), methods[0])
    m = max(r["m"] or 2 for r in rows)
    pairs = [p for p in common.all_length_pairs(m) if any(r["length_pair"] == p for r in rows)]
    groups = {"all": group_summary(rows, cols, methods, reference, select_method)}
    for p in pairs:
        groups[p] = group_summary([r for r in rows if r["length_pair"] == p], cols, methods, reference, select_method)
    summary = dict(run_dir=run_dir, reference=reference, select_method=select_method, select_metric=SELECT_METRIC,
                   degenerate_retention=DEGENERATE_RETENTION,
                   length_standard=common.length_standard(), methods=methods, groups=groups)

    common.write_json(summary, os.path.join(run_dir, "summary.json"))
    cases = sorted({e["case"] for g in groups.values() for e in g["extremes"].values()})
    # a failing render / analysis (Blender crash, full disk, ...) only loses that case's figures, not the report
    if render:
        print(f"rendering the lowest / highest {SELECT_METRIC} case of every group: {cases}")
        from render_results import render_case
        for cid in cases:
            try:
                render_case(run_dir, cid)
            except Exception as e:
                print(f"  {cid}: render FAILED {type(e).__name__}: {e}")
    if analyze and reference in methods and select_method != reference:
        print(f"flux / vector field analysis of {cases}")
        from analysis import analyze_case
        for cid in cases:
            try:
                analyze_case(run_dir, cid, reference, select_method)
            except Exception as e:
                print(f"  {cid}: field analysis FAILED {type(e).__name__}: {e}")

    key_cols = [c for c, _ in KEY_METRICS if any(c in groups["all"]["methods"][mt]["stats"] for mt in methods)]
    lines = [f"# Benchmark summary: {run_dir}", "",
             "Length classes (median strand length): " + ", ".join(_class_str(d) for d in summary["length_standard"]),
             f"Degenerate = min_head_retention < {DEGENERATE_RETENTION} (a head practically vanished); "
             "counted in the averages, picked as lowest / highest only when a group has nothing else."]
    for name, g in groups.items():
        lines += ["", f"## {'All cases' if name == 'all' else name} ({g['n_cases']} cases)", ""]
        lines += _md_tables(g, methods, key_cols, reference)
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    path = report.make_report(run_dir, summary, KEY_METRICS)
    print(f"report: {path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--reference", default="mask_only")
    p.add_argument("--select_method", default="tsai_joint_fm",
                   help="method whose grid score picks each group's lowest / highest case")
    p.add_argument("--no_render", dest="render", action="store_false",
                   help="do not render the lowest / highest cases (the report shows existing renders only)")
    p.add_argument("--no_analysis", dest="analyze", action="store_false",
                   help="do not run the flux / vector field analysis of the lowest / highest cases")
    a = p.parse_args()
    summarize(os.path.abspath(a.run_dir), a.reference, a.select_method, a.render, a.analyze)
