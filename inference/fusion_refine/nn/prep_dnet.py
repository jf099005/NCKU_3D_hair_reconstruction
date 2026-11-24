"""Inputs of the amortised displacement model (disp_net.py) from finished benchmark cases -> one npz per case:
the kept strands' subsampled points per head (P_0, P_1) and their descriptors (sf_0, sf_1, from
features.strand_features with the keep). keep = the --keep_from method's maps.npz (default the FM teacher).

    bash run.sh prep_dnet.py <run_dir> [...] --out $DATA/prep_dnet/<name> [--keep_from afm_a2_guard0.7_slab1]
"""
import argparse
import json
import os
import time

import numpy as np

import common
import diffopt as DO
import features as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--keep_from", default="comb20_slab1", help="method whose maps.npz keep is used")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for rd in a.run_dirs:
        cfg = json.load(open(os.path.join(rd, "run_config.json")))
        bench_dir = os.path.dirname(os.path.dirname(os.path.abspath(rd)))
        for cid in sorted(os.listdir(os.path.join(rd, "cases"))):
            mp_path = os.path.join(rd, "cases", cid, a.keep_from, "maps.npz")
            tag = f"{os.path.basename(bench_dir)}__{cid}"
            out = os.path.join(a.out, f"{tag}.npz")
            if not os.path.isfile(mp_path) or os.path.isfile(out):
                continue
            t0 = time.time()
            case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
            heads = [common.load_hairstyle(h["npz_path"], cfg.get("max_strands"), seed=case["seed"] + i)
                     for i, h in enumerate(case["heads"])]
            labels = common.masks_to_labels([common.load_mask_png(os.path.join(bench_dir, "cases", cid, f),
                                                                  case["mask_res"]) for f in case["mask_files"]])
            mp = np.load(mp_path)
            keep = [mp["keep_0"], mp["keep_1"]]
            fe = F.features(heads, labels, keep=keep)
            sf = F.strand_features(heads, labels, fe, keep=keep)
            np.savez_compressed(out, **{f"P_{i}": h["positions"][k][:, ::DO.STEP].astype(np.float32)
                                        for i, (h, k) in enumerate(zip(heads, keep))},
                                **{f"sf_{i}": s[k].astype(np.float32) for i, (s, k) in enumerate(zip(sf, keep))})
            print(f"{tag}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
