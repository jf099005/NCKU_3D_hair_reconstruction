"""Train the per-strand keep model (keep_model.py) on prep.py npz files (needs sf_* / tkeep_*).

    bash run.sh train_keep.py $DATA/runs/<name> --train <prep dirs> --val <prep dirs> [--epochs 60 --w_flip 10]

Target: flip = teacher keep != drawn split, BCE with flips weighted by w_flip. Left-right flip augmentation of the
texture features (and of the strand descriptors: col coordinate, t_col shape components).
Writes config.json, best.pt (best val F1 of the flips at threshold 0.5), train_log.jsonl.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn

import features as F
import keep_model as KM

N_MAX = 12000   # strands per head per sample in a batch (padded / subsampled)


def strand_flip_signs():
    s = np.ones(len(F.STRAND_FEAT_NAMES), np.float32)
    for k, n in enumerate(F.STRAND_FEAT_NAMES):
        if n.startswith("shape") and n.endswith("_x"):
            s[k] = -1
    return s


def load(dirs):
    items = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            sf = np.concatenate([z["sf_0"], z["sf_1"]]).astype(np.float32)
            uv = np.concatenate([z["uv_0"], z["uv_1"]])
            tk = np.concatenate([z["tkeep_0"], z["tkeep_1"]])
            items.append(dict(x=z["x"].astype(np.float32), sf=sf, uv=uv, flip=(tk != (sf[:, 1] > 0.5)),
                              name=os.path.basename(f)))
    return items


def batch(items, idx, sc, ssc, smu, rng, aug, dev):
    xs, sfs, uvs, ys, ms = [], [], [], [], []
    fs, sfs_sign = F.flip_signs(), strand_flip_signs()
    for i in idx:
        it = items[i]
        n = len(it["sf"])
        sel = rng.choice(n, N_MAX * 2, replace=False) if n > N_MAX * 2 else np.arange(n)
        x, sf, uv = it["x"] / sc[:, None, None], it["sf"][sel].copy(), it["uv"][sel].copy()
        if aug and rng.random() < 0.5:
            x = (x * fs[:, None, None])[:, :, ::-1]
            sf = sf * sfs_sign
            sf[:, 3] = 1 - sf[:, 3]
            uv[:, 1] = 1 - uv[:, 1]
        pad = N_MAX * 2 - len(sel)
        m = np.r_[np.ones(len(sel)), np.zeros(pad)]
        sfn = (sf - smu) / ssc
        xs.append(np.ascontiguousarray(x))
        sfs.append(np.pad(sfn, ((0, pad), (0, 0))))
        uvs.append(np.pad(uv, ((0, pad), (0, 0)), constant_values=0.5))
        ys.append(np.pad(it["flip"][sel].astype(np.float32), (0, pad)))
        ms.append(m)
    t = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(dev)
    return t(xs), t(sfs), t(uvs), t(ys), t(ms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_flip", type=float, default=10.0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    tr, va = load(a.train), load(a.val)
    sc = np.ones(tr[0]["x"].shape[0], np.float32)
    allx = np.stack([it["x"] for it in tr])
    for c in range(4, len(sc)):
        v = allx[:, c]
        sc[c] = max(float(np.sqrt((v[v != 0] ** 2).mean())) if (v != 0).any() else 1.0, 1e-3)
    allsf = np.concatenate([it["sf"][::10] for it in tr])
    smu, ssc = allsf.mean(0), np.maximum(allsf.std(0), 1e-3)
    smu[[0, 1]], ssc[[0, 1]] = 0, 1          # head id / drawn_own stay binary
    smu[[2, 3]], ssc[[2, 3]] = 0.5, 0.5      # uv -> [-1, 1] (flip = 1 - u stays consistent)
    smu[[k for k, n in enumerate(F.STRAND_FEAT_NAMES) if n.startswith("shape") and n.endswith("_x")]] = 0
    cfg = dict(kind="keep", c_feat=len(sc), d_strand=len(smu), width=a.width, ctx=32, hidden=128,
               feat_scale=sc.tolist(), sf_mean=smu.tolist(), sf_scale=ssc.tolist(), feat_res=int(tr[0]["x"].shape[-1]),
               w_flip=a.w_flip, train=a.train, val=a.val, n_train=len(tr), n_val=len(va), epochs=a.epochs, lr=a.lr,
               flip_rate_train=float(np.mean([it["flip"].mean() for it in tr])))
    json.dump(cfg, open(os.path.join(a.run_dir, "config.json"), "w"), indent=1)
    net = KM.build(cfg).to(dev)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, train {len(tr)}, val {len(va)}, "
          f"flip rate {cfg['flip_rate_train']:.4f}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    logf = open(os.path.join(a.run_dir, "train_log.jsonl"), "w")
    best = -1
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(len(tr))
        t0, tl = time.time(), []
        for i in range(0, len(tr) - a.batch + 1, a.batch):
            x, sf, uv, y, m = batch(tr, perm[i:i + a.batch], sc, ssc, smu, rng, True, dev)
            logit = net(x, sf, uv)
            w = m * (1 + (a.w_flip - 1) * y)
            loss = (Fn.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(loss.item())
        net.eval()
        tp = fp = fn = 0
        vl = []
        with torch.no_grad():
            for i in range(0, len(va), a.batch):
                x, sf, uv, y, m = batch(va, range(i, min(i + a.batch, len(va))), sc, ssc, smu, rng, False, dev)
                logit = net(x, sf, uv)
                w = m * (1 + (a.w_flip - 1) * y)
                vl.append(((Fn.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()).item())
                pr = (logit > 0) & (m > 0)
                yt = (y > 0) & (m > 0)
                tp += (pr & yt).sum().item()
                fp += (pr & ~yt).sum().item()
                fn += (~pr & yt).sum().item()
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        r = dict(ep=ep, train=float(np.mean(tl)), val=float(np.mean(vl)), prec=prec, rec=rec, f1=f1,
                 pred_flips=tp + fp, true_flips=tp + fn, sec=round(time.time() - t0, 1))
        logf.write(json.dumps(r) + "\n")
        logf.flush()
        print(json.dumps(r), flush=True)
        torch.save(net.state_dict(), os.path.join(a.run_dir, "last.pt"))
        if f1 > best:
            best = f1
            torch.save(net.state_dict(), os.path.join(a.run_dir, "best.pt"))


if __name__ == "__main__":
    main()
