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
        heads_mt = heads
        ed = os.path.join(od, "edited_heads.npz")
        if os.path.exists(ed):   # strand-editing method: render its edited positions
            e = np.load(ed)
            heads_mt = [dict(h, positions=e[f"positions_{k}"].astype(np.float32)) for k, h in enumerate(heads)]
        print(f"  {cid}/{mt}: rendering")
        ok = render_fused(heads_mt, keep, od, blender_path, keep_blend) and os.path.exists(png)
        out[mt] = png if ok else None
        res_path = os.path.join(od, "result.json")
        res = common.read_json(res_path)
        res["rendered"] = bool(ok)
        common.write_json(res, res_path)
    return out


def render_case_optical(run_dir, cid, methods=None, overwrite=False, device=None):
    """highlighting's optical render (strands_editing.pipeline.render, preset "optical": every strand at the
    hairstyle's metadata radius, Chiang hair BSDF in melanin colors, 32 bounces; base = OPTICAL_COLORS base (dark
    brown), donor = its highlight (golden blonde)) of each method's fused hair -> <method>/render_optical/.
    Meant for full-strand runs (the preset assumes every strand). -> {method: png dir or None}"""
    import json
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    case = common.read_json(os.path.join(cfg["benchmark_dir"], "cases", cid, "case.json"))
    case_dir = os.path.join(run_dir, "cases", cid)
    heads = [common.load_hairstyle(h["npz_path"], cfg.get("max_strands"), seed=case["seed"] + k)
             for k, h in enumerate(case["heads"])]
    normals = [np.load(h["npz_path"])["root_normal"][hh["strand_idx"]] for h, hh in zip(case["heads"], heads)]
    meta = json.load(open(os.path.join(os.path.dirname(case["heads"][0]["npz_path"]), "metadata.json")))
    out = {}
    for mt in sorted(os.listdir(case_dir)):
        od = os.path.join(case_dir, mt)
        if (methods and mt not in methods) or not os.path.exists(os.path.join(od, "maps.npz")):
            continue
        rd = os.path.join(od, "render_optical")
        if os.path.isdir(rd) and any(f.endswith(".png") for f in os.listdir(rd)) and not overwrite:
            out[mt] = rd
            continue
        d = np.load(os.path.join(od, "maps.npz"))
        keep = [d[f"keep_{k}"] for k in range(len(heads))]
        pos = [h["positions"] for h in heads]
        ed = os.path.join(od, "edited_heads.npz")
        if os.path.exists(ed):
            e = np.load(ed)
            pos = [e[f"positions_{k}"].astype(np.float32) for k in range(len(heads))]
        P = np.concatenate([p[k] for p, k in zip(pos, keep)])
        labels = np.concatenate([np.full(int(k.sum()), i > 0) for i, k in enumerate(keep)])
        ruv = np.concatenate([h["root_uv"][k] for h, k in zip(heads, keep)]).astype(np.float32)
        rn = np.concatenate([n[k] for n, k in zip(normals, keep)]).astype(np.float32)
        print(f"  {cid}/{mt}: optical render ({P.shape[0]} strands)")
        try:
            try:
                sec = pl.render(rd, P, labels, ruv, rn, preset="optical", meta=meta, device=device)
            except RuntimeError:
                if device == "CUDA":
                    raise
                # Blender 5.0 OptiX "Misaligned address" on some hair scenes: retry with CUDA
                print("    OptiX failed, retrying with CYCLES_DEVICE=CUDA")
                sec = pl.render(rd, P, labels, ruv, rn, preset="optical", meta=meta, device="CUDA")
            out[mt] = rd
            print(f"    {sec}s")
        except RuntimeError as e:
            print(f"    failed: {e}")
            out[mt] = None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--cases", required=True, help="comma list of case ids, or 'all'")
    p.add_argument("--methods", default=None, help="comma list of methods (default: all in the run)")
    p.add_argument("--overwrite", action="store_true", help="re-render results that already have a render")
    p.add_argument("--keep_blend", action="store_true", help="keep the .blend / fused npz after rendering")
    p.add_argument("--blender_path", default=common.BLENDER_PATH)
    p.add_argument("--optical", action="store_true",
                   help="highlighting's optical render (render_case_optical) instead of the default white/black one")
    p.add_argument("--device", default=None, help="with --optical: CYCLES_DEVICE (CUDA if OptiX crashes)")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cases = (sorted(os.listdir(os.path.join(run_dir, "cases"))) if args.cases == "all"
             else [c.strip() for c in args.cases.split(",")])
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    for i, cid in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {cid}")
        if args.optical:
            render_case_optical(run_dir, cid, methods, args.overwrite, args.device)
        else:
            render_case(run_dir, cid, methods, args.overwrite, args.keep_blend, args.blender_path)


if __name__ == "__main__":
    main()
