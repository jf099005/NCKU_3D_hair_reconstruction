"""Train disp_net.DispNet by minimising diffopt.objective on prep_dnet.py cases (unsupervised).

    bash run.sh train_dnet.py $DATA/runs/<name> --train <prep_dnet dirs> --val <prep_dnet dirs>
        [--epochs 30 --n_pass 2 --w_mix 10 --hidden 256 --lr 1e-3]

Validation: objective / mixing of the prediction vs. no displacement (and vs. diffopt.optimize on the first val
cases, the per-case optimum the model amortises). best.pt = lowest mean val objective.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import diffopt as DO
import disp_net as DN
import features as F


def load(dirs):
    items = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            items.append(dict(P=[z["P_0"], z["P_1"]], sf=[z["sf_0"], z["sf_1"]], name=os.path.basename(f)))
    return items


def to_state(it, cfg, dev):
    st = DO.prepare(it["P"], cfg["obj"], dev)
    mu, sc = np.array(cfg["sf_mean"], np.float32), np.array(cfg["sf_scale"], np.float32)
    return st, [torch.tensor((s - mu) / sc, dtype=torch.float32, device=dev) for s in it["sf"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_pass", type=int, default=2)
    p.add_argument("--w_mix", type=float, default=10.0)
    p.add_argument("--n_opt_ref", type=int, default=6, help="val cases also solved with diffopt.optimize")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    tr, va = load(a.train), load(a.val)
    allsf = np.concatenate([np.concatenate(it["sf"])[::5] for it in tr])
    mu, sc = allsf.mean(0), np.maximum(allsf.std(0), 1e-3)
    mu[[0, 1]], sc[[0, 1]] = 0, 1
    cfg = dict(kind="dnet", n_pt=len(DO.interp_matrix()), d_strand=int(allsf.shape[1]), n_knots=len(F.KNOT_IDX),
               hidden=a.hidden, n_pass=a.n_pass, obj=dict(w_mix=a.w_mix), sf_mean=mu.tolist(), sf_scale=sc.tolist(),
               train=a.train, val=a.val, n_train=len(tr), n_val=len(va), epochs=a.epochs, lr=a.lr)
    json.dump(cfg, open(os.path.join(a.run_dir, "config.json"), "w"), indent=1)
    net = DN.build(cfg).to(dev)
    print(f"{sum(q.numel() for q in net.parameters()) / 1e6:.2f} M params, train {len(tr)}, val {len(va)}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tr))
    # reference: per-case optimum on a few val cases
    ref = {}
    with torch.enable_grad():
        for it in va[:a.n_opt_ref]:
            st, _ = to_state(it, cfg, dev)
            K = cfg["n_knots"]
            D = [torch.zeros(len(q), K, 3, device=dev, requires_grad=True) for q in st["pts"]]
            o = torch.optim.Adam(D, lr=DO.DEFAULTS["lr"])
            for _ in range(DO.DEFAULTS["iters"]):
                loss, _ = DO.objective(D, st)
                o.zero_grad()
                loss.backward()
                o.step()
            ref[it["name"]] = float(DO.objective(D, st)[0])
    logf = open(os.path.join(a.run_dir, "train_log.jsonl"), "w")
    best = 1e9
    for ep in range(a.epochs):
        net.train()
        t0, tl = time.time(), []
        for i in rng.permutation(len(tr)):
            st, sf = to_state(tr[i], cfg, dev)
            loss, parts = DO.objective(net(st, sf), st)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(float(loss))
        net.eval()
        vl, v0, vm, vm0, gap = [], [], [], [], []
        with torch.no_grad():
            for it in va:
                st, sf = to_state(it, cfg, dev)
                D = net(st, sf)
                l, pa = DO.objective(D, st)
                l0, pa0 = DO.objective([torch.zeros_like(d) for d in D], st)
                vl.append(float(l)), v0.append(float(l0)), vm.append(float(pa["mix"])), vm0.append(float(pa0["mix"]))
                if it["name"] in ref:
                    gap.append((float(l) - ref[it["name"]]) / max(float(l0) - ref[it["name"]], 1e-9))
        r = dict(ep=ep, train=float(np.mean(tl)), val=float(np.mean(vl)), val_zero=float(np.mean(v0)),
                 mix=float(np.mean(vm)), mix_zero=float(np.mean(vm0)),
                 gap_to_opt=float(np.mean(gap)) if gap else None, sec=round(time.time() - t0, 1))
        logf.write(json.dumps(r) + "\n")
        logf.flush()
        print(json.dumps(r), flush=True)
        torch.save(net.state_dict(), os.path.join(a.run_dir, "last.pt"))
        if r["val"] < best:
            best = r["val"]
            torch.save(net.state_dict(), os.path.join(a.run_dir, "best.pt"))


if __name__ == "__main__":
    main()
