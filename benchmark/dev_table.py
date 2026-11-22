"""Quick per-case / per-method table of a dev run (adaptive_fm experiments).

    python benchmark/dev_table.py <run_dir> [--methods a,b] [--per_case]

per case: gs/ret/edge_frac (or /lam without view_mix). ret = min over heads of kept strands / strands the drawn mask gives that head (mask_only's kept);
gs = grid_score.mean_all; bald = coverage.bald_frac. Mean over the cases where every listed method is ok.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(run_dir):
    out = {}
    for p in glob.glob(os.path.join(run_dir, "cases", "*", "*", "result.json")):
        j = json.load(open(p))
        out.setdefault(j["case"], {})[j["method"]] = j
    return out


def row(j, ref):
    if j is None or j.get("status") != "ok":
        return None
    m = j["metrics"]
    kept, kept0 = m["strands"]["kept"], ref["metrics"]["strands"]["kept"]
    ret = min(k / max(k0, 1) for k, k0 in zip(kept, kept0))
    vm = m.get("view_mix", {})
    return dict(gs=m["grid_score"]["mean_all"], ret=ret, bald=m["coverage"]["bald_frac"],
                edge=vm.get("edge_frac", np.nan), blk=vm.get("block_mix", np.nan), dvis=vm.get("donor_vis", np.nan),
                pairs=m["crossings"]["pairs"], lam=(j.get("method_info") or {}).get("lam"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--per_case", action="store_true")
    a = ap.parse_args()
    res = load(a.run_dir)
    methods = a.methods.split(",") if a.methods else sorted({m for r in res.values() for m in r},
                                                            key=lambda s: (s != "mask_only", s != "tsai_joint_fm", s))
    rows = {}
    for cid in sorted(res):
        ref = res[cid].get("mask_only")
        if ref is None:
            continue
        rows[cid] = {mt: row(res[cid].get(mt), ref) for mt in methods}
    if a.per_case:
        print("case".ljust(10) + "".join(mt[:18].rjust(20) for mt in methods))
        for cid, r in rows.items():
            print(cid.ljust(10) + "".join(
                (f"{v['gs']:.3f}/{v['ret']:.2f}" + (f"/{v['edge']:.3f}" if np.isfinite(v['edge']) else (f"/{v['lam']}" if v['lam'] else ""))).rjust(20) if v else "-".rjust(20)
                for v in r.values()))
    full = [cid for cid, r in rows.items() if all(r[mt] for mt in methods)]
    print(f"\nmean over {len(full)} cases with every method ok")
    print("method".ljust(22) + "gs".rjust(8) + "ret".rjust(8) + "ret_min".rjust(8) + "#ret<.5".rjust(8)
          + "bald".rjust(8) + "gs_win".rjust(8) + "edge".rjust(8) + "blkmix".rjust(8) + "dvis".rjust(8))
    for mt in methods:
        v = [rows[c][mt] for c in full]
        if not v:
            continue
        gs = np.array([x["gs"] for x in v]); ret = np.array([x["ret"] for x in v])
        gs0 = np.array([rows[c]["mask_only"]["gs"] for c in full]) if "mask_only" in methods else gs
        print(mt.ljust(22) + f"{gs.mean():8.4f}{ret.mean():8.3f}{ret.min():8.3f}{int((ret < .5).sum()):8d}"
              f"{np.mean([x['bald'] for x in v]):8.4f}{(gs < gs0).mean():8.2f}"
              f"{np.mean([x['edge'] for x in v]):8.4f}{np.mean([x['blk'] for x in v]):8.4f}{np.mean([x['dvis'] for x in v]):8.3f}")


if __name__ == "__main__":
    main()
