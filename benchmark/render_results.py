#!/usr/bin/env python3
"""Render fused results of an existing benchmark run, without rerunning the fusion.

    python benchmark/render_results.py <run_dir> --cases case_0000,case_0003 [--methods tsai_joint_fm]
           [--overwrite] [--keep_blend]

summarize.py already renders the cases the report needs (lowest / highest grid score of every hair-length
pair); use this for any other case. run_benchmark.py saves every method's per-head keep masks in
<case>/<method>/maps.npz (keep_0, keep_1, ...). This reloads the case's hairstyles exactly as the run did
(same max_strands and per-head seed, from run_config.json), composes the fused hair from those masks and
renders it: <case>/<method>/render/render_multiview.png (base white, donor black). Existing renders are
reused unless --overwrite.
"""
import argparse
import os

import numpy as np

import common


def render_path(run_dir, cid, method):
    return os.path.join(run_dir, "cases", cid, method, "render", "render_multiview.png")


def render_case(run_dir, cid, methods=None, overwrite=False, keep_blend=False, blender_path=common.BLENDER_PATH):
    """Render the given methods (default: all) of one case. Returns {method: render png path or None}."""
    from run_benchmark import render_fused   # imports img2hair_tsai_1 lazily
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    case = common.read_json(os.path.join(cfg["benchmark_dir"], "cases", cid, "case.json"))
    case_dir = os.path.join(run_dir, "cases", cid)
    out, todo = {}, []
    for mt in sorted(os.listdir(case_dir)):
        # method folders only (a case also has analysis/ from summarize.py)
        if (methods and mt not in methods) or not os.path.exists(os.path.join(case_dir, mt, "result.json")):
            continue
        od = os.path.join(case_dir, mt)
        png = render_path(run_dir, cid, mt)
        if not os.path.exists(os.path.join(od, "maps.npz")):
            print(f"  {cid}/{mt}: no maps.npz (method failed?), not rendered")
            out[mt] = None
        elif os.path.exists(png) and not overwrite:
            out[mt] = png
        else:
            todo.append((mt, od, png))
    if not todo:
        return out

    # same strands as the run: max_strands subset with seed case_seed + head index
    heads = [common.load_hairstyle(h["npz_path"], cfg.get("max_strands"), seed=case["seed"] + k)
             for k, h in enumerate(case["heads"])]
    for mt, od, png in todo:
        d = np.load(os.path.join(od, "maps.npz"))
        keep = [d[f"keep_{k}"] for k in range(len(heads))]
        if any(k.size != h["root_uv"].shape[0] for k, h in zip(keep, heads)):
            print(f"  {cid}/{mt}: keep masks do not match the hairstyles' strand counts "
                  "(dataset or max_strands changed?), not rendered")
            out[mt] = None
            continue
        print(f"  {cid}/{mt}: rendering")
        ok = render_fused(heads, keep, od, blender_path, keep_blend) and os.path.exists(png)
        out[mt] = png if ok else None
        res_path = os.path.join(od, "result.json")
        res = common.read_json(res_path)
        res["rendered"] = bool(ok)
        common.write_json(res, res_path)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--cases", required=True, help="comma list of case ids, or 'all'")
    p.add_argument("--methods", default=None, help="comma list of methods (default: all in the run)")
    p.add_argument("--overwrite", action="store_true", help="re-render results that already have a render")
    p.add_argument("--keep_blend", action="store_true", help="keep the .blend / fused npz after rendering")
    p.add_argument("--blender_path", default=common.BLENDER_PATH)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cases = (sorted(os.listdir(os.path.join(run_dir, "cases"))) if args.cases == "all"
             else [c.strip() for c in args.cases.split(",")])
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    for i, cid in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {cid}")
        render_case(run_dir, cid, methods, args.overwrite, args.keep_blend, args.blender_path)


if __name__ == "__main__":
    main()
