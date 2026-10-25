#!/usr/bin/env python3
"""Run fusion methods on a benchmark built by build_benchmark.py and score them.

    python benchmark/run_benchmark.py <benchmark_dir> [--run_name run] [--methods mask_only,tsai_joint_fm]
           [--cases case_0000,case_0003] [--max_strands N] [--metrics strands,coverage,crossings,grid_score,flux]
           [--grid_n 32] [--no_render] [--overwrite] [--max_mem_gb G] [--min_free_gb 4] [--no_isolate]
           [--tsai2 [--trim_percent 0.5]] [--boundary_fm]

Per case (img2hair_tsai_1.py's pipeline, with the dataset hairstyles in place of the hairstyles DiffLocks
reconstructs from the images): load the m hairstyles (root_uv converted to the scalp-texture layout),
run every method on the case's label map, compose the fused hair from the kept strands, score it.

    <benchmark_dir>/runs/<run_name>/run_config.json
    .../cases/<case_id>/<method>/result.json   status, kept strands, method info (FM objectives, ...), metrics
    .../cases/<case_id>/<method>/maps.npz      per metric (n, n) maps (grid score, |flux|, crossings, bald cells)
    .../cases/<case_id>/<method>/labels_refined.png, roots.png
    .../cases/<case_id>/<method>/render/       multi-view render (base white / donor black), only for the lowest /
                                               highest grid-score case of every hair-length pair (summarize.py)
    .../summary.csv, summary.json, summary.md, report.html   (summarize.py; rerun it alone any time)

--tsai2 also runs inference/img2hair_tsai_2.py's pipeline as method "tsai2_joint_fm_trim": the joint FM of
tsai_joint_fm, then trimming of --trim_percent % of all strands (the visible ones crossing the other head the
most). The FM then runs once per case and both tsai methods share it (same result as without --tsai2).
Without --tsai2 nothing changes.

--boundary_fm runs the joint FM of the tsai methods as Boundary FM (JOINT_FM_KWARGS + boundary=True,
joint_contour.boundary_fm). run_config.json pins it: compare with plain FM in a separate --run_name.
Without --boundary_fm nothing changes.

Finished (case, method) pairs are skipped on a rerun unless --overwrite, so a run can be resumed.

Memory guard: every case runs in its own child process (its memory is returned after the case) that the
parent watches. If the child's RSS passes --max_mem_gb (default 70% of the RAM) or the free system memory
drops under --min_free_gb, the child is killed, the method it was on gets result.json status "skipped_oom"
and the run moves on (the case's other methods are rerun in a new child). A rerun skips skipped_oom results
too; use --retry_skipped to try them again (e.g. with --max_strands). The child also raises its own
oom_score_adj, so if the kernel OOM killer still fires it takes the benchmark, not the desktop / editor.
"""
import argparse
import gc
import os
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

import common
import fusion_methods
import metrics as metrics_mod
import summarize


TSAI2 = "tsai2_joint_fm_trim"


def save_roots_figure(heads, keep, labels, path, title):
    """kept roots of every head in its color, dropped ones gray, over the label map (scalp-texture layout)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 5))
    for i, (ax, h, k) in enumerate(zip(np.atleast_1d(axes), heads, keep)):
        ax.imshow(labels, cmap="Greys", vmin=0, vmax=max(int(labels.max()), 1) * 4, extent=(0, 1, 1, 0))
        uv = h["root_uv"]
        ax.scatter(uv[~k, 1], uv[~k, 0], s=0.15, linewidths=0, c="#a3a29c")
        ax.scatter(uv[k, 1], uv[k, 0], s=0.15, linewidths=0, c=common.HEAD_COLORS[i % len(common.HEAD_COLORS)])
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        ax.set_title(f"head {i}: kept {int(k.sum())}/{k.size}")
    fig.suptitle(title)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def strand_colors(m, i):
    """per-head render color: m = 2 as img2hair_tsai_1 (base white, donor black), else a gray ramp"""
    if m == 2:
        import img2hair_tsai_1 as tsai
        return (tsai.BASE_STRAND_COLOR, tsai.DONOR_STRAND_COLOR)[i]
    g = 1.0 - i / (m - 1)
    return (g, g, g)


def render_fused(heads, keep, out_dir, blender_path, keep_blend):
    """compose the fused npz (img2hair_tsai_1.compose_hair_per_strand's format, any m), build the .blend
    with img2hair_tsai_1.create_blender_file and render it with experiments/gpu_render.py"""
    import img2hair_tsai_1 as tsai
    m = len(heads)
    npz_path = os.path.join(out_dir, "mixed_output_strands.npz")
    np.savez(npz_path,
             positions=np.concatenate([h["positions"][k] for h, k in zip(heads, keep)]).astype(np.float32),
             root_uv=np.concatenate([h["root_uv"][k] for h, k in zip(heads, keep)]).astype(np.float32),
             strand_colors=np.concatenate([np.tile(np.float32(strand_colors(m, i)), (int(k.sum()), 1))
                                           for i, k in enumerate(keep)]),
             source=np.concatenate([np.full(int(k.sum()), i, np.int8) for i, k in enumerate(keep)]))
    args = lambda: None
    args.blender_path = blender_path
    blend = tsai.create_blender_file(args, out_dir, npz_path)
    ok = False
    if blend is not None:
        r = subprocess.run([blender_path, "-b", blend, "--python",
                            os.path.join(common.REPO_ROOT, "experiments", "gpu_render.py"),
                            "--", "--out_path", os.path.join(out_dir, "render")])
        ok = r.returncode == 0
    if not keep_blend:
        for fn in ("mixed_output_strands.npz", "blender_scene.blend", "blender_scene.blend1"):
            p = os.path.join(out_dir, fn)
            if os.path.exists(p):
                os.remove(p)
    return ok


def needs_run(result_path, args):
    """no result yet, --overwrite, or a skipped_oom result with --retry_skipped"""
    if args.overwrite or not os.path.exists(result_path):
        return True
    return args.retry_skipped and common.read_json(result_path).get("status") == "skipped_oom"


def _meminfo_gb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def _rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def worker_argv(args, run_dir, cid, methods):
    """command line of the child that runs `methods` of one case (same settings, no run_config / summary).
    The parent already decided what to (re)run, so the child always overwrites."""
    argv = [sys.executable, os.path.abspath(__file__), args.benchmark_dir, "--out", run_dir, "--cases", cid,
            "--methods", ",".join(methods), "--overwrite", "--metrics", ",".join(args.metrics),
            "--grid_n", str(args.grid_n), "--trim_percent", str(args.trim_percent), "--_worker"]
    if args.boundary_fm:
        argv.append("--boundary_fm")
    if args.max_strands is not None:
        argv += ["--max_strands", str(args.max_strands)]
    for flag, on in (("--no_figures", not args.figures), ("--verbose", args.verbose)):
        if on:
            argv.append(flag)
    return argv


def run_case_guarded(case_dir, run_dir, args):
    """run_case in a child process under the memory guard; on a kill mark the running method skipped_oom and
    restart the child for the case's remaining methods"""
    case = common.read_json(os.path.join(case_dir, "case.json"))
    cid = case["id"]
    methods = [mt for mt in args.methods if fusion_methods.method_supports(mt, case["m"])]
    res_path = lambda mt: os.path.join(run_dir, "cases", cid, mt, "result.json")
    attempted = set()
    while True:
        todo = [mt for mt in methods if mt not in attempted and needs_run(res_path(mt), args)]
        if not todo:
            if not attempted:
                print(f"  {cid}: done already")
            return
        before = {mt: os.path.getmtime(res_path(mt)) if os.path.exists(res_path(mt)) else None for mt in todo}
        proc = subprocess.Popen(worker_argv(args, run_dir, cid, todo), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        peak, reason = 0.0, None
        while proc.poll() is None:
            rss, free = _rss_gb(proc.pid), _meminfo_gb("MemAvailable")
            peak = max(peak, rss)
            if rss > args.max_mem_gb:
                reason = f"RSS {rss:.1f} GB > --max_mem_gb {args.max_mem_gb:.1f}"
            elif free < args.min_free_gb:
                reason = f"free system memory {free:.1f} GB < --min_free_gb {args.min_free_gb:.1f} (RSS {rss:.1f} GB)"
            if reason:
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                break
            time.sleep(0.25)
        if reason is None and proc.returncode == -signal.SIGKILL:
            reason = f"killed by the kernel OOM killer (RSS peak {peak:.1f} GB)"
        written = [mt for mt in todo if os.path.exists(res_path(mt)) and
                   (before[mt] is None or os.path.getmtime(res_path(mt)) > before[mt])]
        for mt in written:
            res = common.read_json(res_path(mt))
            res["case_peak_rss_gb"] = round(peak, 2)
            common.write_json(res, res_path(mt))
        attempted.update(written)
        if reason is None and proc.returncode == 0:
            continue                                       # all remaining methods done (the loop returns)
        # the first method without a fresh result is the one that was running
        mt = next(mt for mt in todo if mt not in written)
        attempted.add(mt)
        status = "skipped_oom" if reason else "failed"
        error = reason or f"worker exited with code {proc.returncode}"
        print(f"    {mt}: {status.upper()} {error}")
        os.makedirs(os.path.dirname(res_path(mt)), exist_ok=True)
        common.write_json(dict(case=cid, method=mt, m=case["m"], heads=[h["name"] for h in case["heads"]],
                               mask_generator=case["mask_generator"], max_strands=args.max_strands,
                               status=status, error=error, case_peak_rss_gb=round(peak, 2),
                               max_mem_gb=args.max_mem_gb, finished=common.now_str()), res_path(mt))
        gc.collect()


def run_case(case_dir, run_dir, args, ctx):
    case = common.read_json(os.path.join(case_dir, "case.json"))
    cid, m = case["id"], case["m"]
    methods = [mt for mt in args.methods if fusion_methods.method_supports(mt, m)]
    skipped = [mt for mt in args.methods if mt not in methods]
    if skipped:
        print(f"  {cid}: methods {skipped} do not support m = {m}, skipped")
    out_dirs = {mt: os.path.join(run_dir, "cases", cid, mt) for mt in methods}
    todo = [mt for mt in methods if needs_run(os.path.join(out_dirs[mt], "result.json"), args)]
    if not todo:
        print(f"  {cid}: done already")
        return

    t0 = time.time()
    heads = [common.load_hairstyle(h["npz_path"], args.max_strands, seed=case["seed"] + i)
             for i, h in enumerate(case["heads"])]
    masks = [common.load_mask_png(os.path.join(case_dir, f), case["mask_res"]) for f in case["mask_files"]]
    labels = common.masks_to_labels(masks)
    load_s = time.time() - t0
    t0 = time.time()
    cc = metrics_mod.CaseContext(heads, args.grid_n, ctx["dataset_path"], ctx)
    init_keep = fusion_methods.initial_keep(heads, labels)
    print(f"  {cid}: {' + '.join(h['name'] for h in case['heads'])} "
          f"({', '.join(str(h['root_uv'].shape[0]) for h in heads)} strands), {case['mask_generator']}; "
          f"load {load_s:.1f}s, membership {time.time() - t0:.1f}s")

    for mt in todo:
        od = out_dirs[mt]
        os.makedirs(od, exist_ok=True)
        res = dict(case=cid, method=mt, m=m, heads=[h["name"] for h in case["heads"]],
                   mask_generator=case["mask_generator"], max_strands=args.max_strands, finished=None)
        try:
            t0 = time.time()
            fr = fusion_methods.METHODS[mt](heads, labels, ctx)
            res["method_seconds"] = round(time.time() - t0, 2)
            res["method_info"] = fr.info
            scalars, maps, timings = metrics_mod.evaluate(cc, fr.keep, init_keep, args.metrics)
            res["metrics"], res["metric_seconds"] = scalars, timings
            np.savez_compressed(os.path.join(od, "maps.npz"), **maps,
                                **{f"keep_{i}": k for i, k in enumerate(fr.keep)})
            if fr.refined_labels is not None:
                common.save_label_preview(fr.refined_labels, None, os.path.join(od, "labels_refined.png"))
            if args.figures:
                save_roots_figure(heads, fr.keep, labels, os.path.join(od, "roots.png"), f"{cid} / {mt}")
            res["status"] = "ok"
            g = scalars.get("grid_score", {})
            c = scalars.get("crossings", {})
            ov = fr.info.get("overlap_matrix")
            print(f"    {mt}: {res['method_seconds']}s, grid score {g.get('mean_all', float('nan')):.4f} "
                  f"(/black {g.get('per_black_area', float('nan')):.4f}, /white {g.get('per_white_area', float('nan')):.4f}), "
                  f"crossing pairs {c.get('pairs', -1)}, kept {scalars.get('strands', {}).get('kept')}"
                  + (f", overlap matrix nnz {ov['nnz']} (density {ov['density']:.2e})" if ov else ""))
        except Exception as e:   # one broken case must not stop the run
            res["status"] = "failed"
            res["error"] = f"{type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()
            print(f"    {mt}: FAILED {res['error']}")
        res["finished"] = common.now_str()
        common.write_json(res, os.path.join(od, "result.json"))

    ctx.pop("_joint_fm_cache", None)   # shared joint FM of this case (tsai2_joint_fm_trim runs)
    del heads, cc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark_dir")
    p.add_argument("--run_name", default="run", help="results go to <benchmark_dir>/runs/<run_name>")
    p.add_argument("--out", default=None, help="explicit results folder (overrides --run_name)")
    p.add_argument("--methods", default="mask_only,tsai_joint_fm",
                   help=f"comma list; known: {','.join(fusion_methods.METHODS)}")
    p.add_argument("--metrics", default=",".join(metrics_mod.DEFAULT_METRICS),
                   help=f"comma list; known: {','.join(metrics_mod.METRICS)}")
    p.add_argument("--cases", default=None, help="comma list of case ids (default: all)")
    p.add_argument("--max_strands", type=int, default=None,
                   help="random subset of strands per head (default: all; fewer = faster, less memory)")
    p.add_argument("--grid_n", type=int, default=32, help="analysis grid of the metrics (as debug_field_analysis)")
    p.add_argument("--no_figures", dest="figures", action="store_false")
    p.add_argument("--no_render", dest="render", action="store_false",
                   help="skip rendering the lowest / highest grid-score case of every length pair at the end")
    p.add_argument("--overwrite", action="store_true", help="recompute finished (case, method) pairs")
    p.add_argument("--verbose", action="store_true", help="progress output of joint_contour")
    total = _meminfo_gb("MemTotal")
    p.add_argument("--max_mem_gb", type=float, default=round(0.7 * total, 1),
                   help=f"kill a case whose process uses more RAM than this (default 70%% of {total:.0f} GB)")
    p.add_argument("--min_free_gb", type=float, default=4.0,
                   help="also kill it when the system's available memory drops below this")
    p.add_argument("--retry_skipped", action="store_true", help="rerun results marked skipped_oom")
    p.add_argument("--no_isolate", dest="isolate", action="store_false",
                   help="run the cases in this process, without the memory guard")
    p.add_argument("--tsai2", action="store_true",
                   help="also run img2hair_tsai_2 (joint FM + trimming) as method tsai2_joint_fm_trim")
    p.add_argument("--trim_percent", type=float, default=0.5,
                   help="tsai2_joint_fm_trim: trim this %% of all strands (base + donor) after the FM (default 0.5)")
    p.add_argument("--boundary_fm", action="store_true",
                   help="run the joint FM of the tsai methods as Boundary FM (default off = plain FM)")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    bench_dir = os.path.abspath(args.benchmark_dir)
    bench = common.read_json(os.path.join(bench_dir, "benchmark.json"))
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    args.metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
    if args.tsai2 and TSAI2 not in args.methods:
        args.methods.append(TSAI2)
    if args.trim_percent < 0:
        p.error("--trim_percent must be >= 0")
    for name, known in (("method", fusion_methods.METHODS), ("metric", metrics_mod.METRICS)):
        bad = [x for x in (args.methods if name == "method" else args.metrics) if x not in known]
        if bad:
            p.error(f"unknown {name}s {bad}; known: {list(known)}")
    cases = bench["cases"] if args.cases is None else [c.strip() for c in args.cases.split(",")]
    run_dir = os.path.abspath(args.out or os.path.join(bench_dir, "runs", args.run_name))
    args.benchmark_dir = bench_dir
    ctx = dict(dataset_path=bench["dataset_path"], verbose=args.verbose)
    if TSAI2 in args.methods:   # only then: shared FM + trimming settings (otherwise ctx is as before)
        ctx.update(share_joint_fm=True, trim_percent=args.trim_percent)
    if args.boundary_fm:   # only then (otherwise ctx is as before)
        ctx["boundary_fm"] = True

    if args._worker:   # child of run_case_guarded: one case, first in line for the kernel OOM killer
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        for cid in cases:
            run_case(os.path.join(bench_dir, "cases", cid), run_dir, args, ctx)
        return

    cfg = dict(benchmark_dir=bench_dir, methods=args.methods, metrics=args.metrics, max_strands=args.max_strands,
               grid_n=args.grid_n, started=common.now_str())
    if TSAI2 in args.methods:
        cfg["trim_percent"] = args.trim_percent
    if args.boundary_fm:
        cfg["boundary_fm"] = True
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        old = common.read_json(cfg_path)
        diff = [k for k in ("max_strands", "grid_n") if old.get(k) != cfg[k]]
        if "trim_percent" in cfg and old.get("trim_percent", cfg["trim_percent"]) != cfg["trim_percent"]:
            diff.append("trim_percent")
        if old.get("boundary_fm", False) != args.boundary_fm:
            diff.append("boundary_fm")
        if diff and not args.overwrite:
            p.error(f"{run_dir} was run with different {diff} ({[old.get(k) for k in diff]}); "
                    "use another --run_name or --overwrite")
    if os.path.exists(cfg_path) and "trim_percent" not in cfg and "trim_percent" in old:
        cfg["trim_percent"] = old["trim_percent"]      # keep the record of an earlier --tsai2 resume
    common.write_json(cfg, cfg_path)

    print(f"benchmark {bench_dir}: {len(cases)} cases, m = {bench['m']}, methods {args.methods} -> {run_dir}")
    if args.isolate:
        print(f"memory guard: kill a case above {args.max_mem_gb:.1f} GB RSS or below {args.min_free_gb:.1f} GB free")
    t_all = time.time()
    for i, cid in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {cid}")
        if args.isolate:
            run_case_guarded(os.path.join(bench_dir, "cases", cid), run_dir, args)
        else:
            run_case(os.path.join(bench_dir, "cases", cid), run_dir, args, ctx)
    print(f"all cases done in {(time.time() - t_all) / 60:.1f} min")
    summarize.summarize(run_dir, render=args.render)


if __name__ == "__main__":
    main()
