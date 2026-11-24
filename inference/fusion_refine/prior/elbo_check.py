"""Feasibility check of the ELBO maps on benchmark cases (full strands): do the maps light up at the fusion
boundary? For each case: original base, original donor, the fusion with the drawn mask (mask_only), and the
fusion with a refined mask (labels_refined.png of --refined_method, if that run exists).

    cd <repo> && bash inference/fusion_refine/nn/run.sh ../prior/elbo_check.py <bench_dir> --cases case_0000,...
        [--refined_run <run_dir> --refined_method comb20_slab1] --out <dir>

Writes per case maps.npz (every variant x sigma), stats.json (mean error in the boundary band vs the rest of the
hair region) and a png (rows: variants, columns: sigmas).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (benchmark)
from fusion_refine._util import boundary, hair_region, keep_from_mask  # noqa: E402
from difflocks_prior import DiffLocksPrior  # noqa: E402

SIGMAS = (0.05, 0.1, 0.3, 1.0)


def refined_mask(path):
    img = cv2.imread(path)
    return (img[..., 2] > 200) & (img[..., 0] < 100)


def band(mask, region, width):
    from scipy.ndimage import binary_dilation
    return binary_dilation(boundary(mask, region), iterations=width) & region


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench_dir")
    p.add_argument("--cases", required=True)
    p.add_argument("--refined_run", default=None)
    p.add_argument("--refined_method", default="comb20_slab1")
    p.add_argument("--out", required=True)
    p.add_argument("--n_noise", type=int, default=4)
    p.add_argument("--band", type=int, default=4, help="boundary band half width (texels)")
    a = p.parse_args()
    prior = DiffLocksPrior()
    os.makedirs(a.out, exist_ok=True)
    allstats = {}
    for cid in a.cases.split(","):
        cdir = os.path.join(a.bench_dir, "cases", cid)
        case = common.read_json(os.path.join(cdir, "case.json"))
        heads = [common.load_hairstyle(h["npz_path"]) for h in case["heads"]]   # all strands
        labels = common.masks_to_labels([common.load_mask_png(os.path.join(cdir, f), case["mask_res"])
                                         for f in case["mask_files"]])
        drawn = labels == 1
        lat = [prior.encode(h["positions"]) for h in heads]                       # encode once, reuse
        region = hair_region([h["root_uv"] for h in heads])
        variants = {"orig_base": [np.ones(len(heads[0]["root_uv"]), bool), np.zeros(len(heads[1]["root_uv"]), bool)],
                    "orig_donor": [np.zeros(len(heads[0]["root_uv"]), bool), np.ones(len(heads[1]["root_uv"]), bool)],
                    "mask_only": keep_from_mask(heads, drawn)}
        masks = {"mask_only": drawn}
        rp = os.path.join(a.refined_run or "", "cases", cid, a.refined_method, "labels_refined.png")
        if a.refined_run and os.path.isfile(rp):
            m = refined_mask(rp)
            variants["refined"] = keep_from_mask(heads, m)
            masks["refined"] = m
        maps, stats = {}, {}
        for name, keep in variants.items():
            pos = np.concatenate([h["positions"][k] for h, k in zip(heads, keep)])
            uv = np.concatenate([h["root_uv"][k] for h, k in zip(heads, keep)])
            z = torch_cat([l[k] for l, k in zip(lat, keep)])
            tex, info = prior.hair_to_texture(pos, uv, latents=z)
            em = prior.elbo_maps(tex, SIGMAS, a.n_noise)
            for s, mp in em.items():
                maps[f"{name}_s{s:g}"] = mp.astype(np.float32)
            bm = band(masks.get(name, drawn), region, a.band)   # originals: evaluated at the drawn boundary
            rest = region & ~bm
            stats[name] = {f"s{s:g}": dict(band=float(mp[bm].mean()), rest=float(mp[rest].mean()),
                                           ratio=float(mp[bm].mean() / max(mp[rest].mean(), 1e-9)))
                           for s, mp in em.items()}
            stats[name]["fill"] = info["fill"]
            print(cid, name, {k: (round(v["band"], 3), round(v["rest"], 3), round(v["ratio"], 2))
                              for k, v in stats[name].items() if k != "fill"}, flush=True)
        np.savez_compressed(os.path.join(a.out, f"{cid}_maps.npz"), region=region, drawn=drawn, **maps)
        allstats[cid] = stats
        save_png(maps, list(variants), region, drawn, os.path.join(a.out, f"{cid}.png"), cid)
    json.dump(allstats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)


def torch_cat(xs):
    import torch
    return torch.cat(xs)


def save_png(maps, names, region, drawn, path, title):
    tiles = []
    for n in names:
        row = []
        for s in SIGMAS:
            mp = maps[f"{n}_s{s:g}"]
            hi = np.percentile(np.concatenate([maps[f"{m}_s{s:g}"][region] for m in names]), 99)
            im = np.clip(mp / max(hi, 1e-9), 0, 1)
            im = cv2.applyColorMap((im * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            im[~region] = 40
            edge = boundary(drawn, region)
            im[edge] = (255, 255, 255)
            cv2.putText(im, f"{n} s={s:g}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            row.append(im)
        tiles.append(np.hstack(row))
    img = np.vstack(tiles)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    main()
