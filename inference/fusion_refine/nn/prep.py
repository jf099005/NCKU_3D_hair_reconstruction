"""Features + teacher targets of finished benchmark cases -> one small npz per (case, strand subset).

    bash run.sh prep.py <run_dir> [<run_dir> ...] --teacher comb20_slab1 --out $DATA/prep/<name> [--subsets 2]

For every case with an ok teacher result: heads are reloaded exactly as the benchmark did (max_strands, seed
case_seed + i), the teacher keep (maps.npz), refined labels (labels_refined.png) and edited positions
(edited_heads.npz, if the teacher moved strands) give the targets. Subset 0 = all strands of the run; subset
k > 0 = a random half of them (augmentation: the features change with the strand sample, the targets barely).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import common
import features as F


def refined_mask(path):
    img = cv2.imread(path)            # BGR; donor = HEAD_COLORS[1] (#eb6834)
    return (img[..., 2] > 200) & (img[..., 0] < 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--teacher", default="comb20_slab1")
    p.add_argument("--out", required=True)
    p.add_argument("--subsets", type=int, default=2)
    p.add_argument("--res", type=int, default=F.FEAT_RES)
    p.add_argument("--keep_feat", action="store_true",
                   help="texture features on the teacher's kept strands (input of a displacement model that runs "
                        "after the keep decision) instead of the drawn split")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for rd in a.run_dirs:
        cfg = json.load(open(os.path.join(rd, "run_config.json")))
        bench_dir = os.path.dirname(os.path.dirname(os.path.abspath(rd)))
        for cid in sorted(os.listdir(os.path.join(rd, "cases"))):
            tdir = os.path.join(rd, "cases", cid, a.teacher)
            rj = os.path.join(tdir, "result.json")
            tag = f"{os.path.basename(bench_dir)}__{cid}"
            if not os.path.isfile(rj) or json.load(open(rj)).get("status", "ok") != "ok":
                continue
            if os.path.isfile(os.path.join(a.out, f"{tag}__s{a.subsets - 1}.npz")):
                continue
            t0 = time.time()
            case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
            heads = [common.load_hairstyle(h["npz_path"], cfg.get("max_strands"), seed=case["seed"] + i)
                     for i, h in enumerate(case["heads"])]
            masks = [common.load_mask_png(os.path.join(bench_dir, "cases", cid, f), case["mask_res"])
                     for f in case["mask_files"]]
            labels = common.masks_to_labels(masks)
            mp = np.load(os.path.join(tdir, "maps.npz"))
            keep = [mp["keep_0"], mp["keep_1"]]
            mask = refined_mask(os.path.join(tdir, "labels_refined.png"))
            from fusion_refine._util import keep_from_mask
            agree = [float((k == k2).mean()) for k, k2 in zip(keep, keep_from_mask(heads, mask))]
            ep = os.path.join(tdir, "edited_heads.npz")
            if os.path.isfile(ep):
                e = np.load(ep)
                edited = [e["positions_0"].astype(np.float32), e["positions_1"].astype(np.float32)]
            else:
                edited = [h["positions"] for h in heads]
            rng = np.random.default_rng(case["seed"])
            for s in range(a.subsets):
                if s == 0:
                    sel = [np.arange(len(h["root_uv"])) for h in heads]
                else:
                    sel = [np.sort(rng.choice(len(h["root_uv"]), len(h["root_uv"]) // 2, replace=False))
                           for h in heads]
                hs = [dict(positions=h["positions"][ix], root_uv=h["root_uv"][ix]) for h, ix in zip(heads, sel)]
                fe = F.features(hs, labels, a.res, keep=[k[ix] for k, ix in zip(keep, sel)] if a.keep_feat else None)
                y, w = F.targets(hs, [k[ix] for k, ix in zip(keep, sel)], mask, [E[ix] for E, ix in zip(edited, sel)], fe)
                sf = F.strand_features(hs, labels, fe) if not a.keep_feat else None
                extra = {}
                for hi in range(2 if sf is not None else 0):
                    extra[f"sf_{hi}"] = sf[hi].astype(np.float16)
                    extra[f"uv_{hi}"] = hs[hi]["root_uv"].astype(np.float32)
                    extra[f"tkeep_{hi}"] = keep[hi][sel[hi]]
                np.savez_compressed(os.path.join(a.out, f"{tag}__s{s}.npz"), x=fe["x"].astype(np.float16),
                                    y=y.astype(np.float32), w=w, region=fe["region"], **extra,
                                    agree=np.array(agree), n_strands=np.array([len(ix) for ix in sel]))
            print(f"{tag}: keep/mask agreement {agree}, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
