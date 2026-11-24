"""Train the scalp-texture displacement model on prep.py npz files (the refined-label channel of the targets is
not used: the keep decision is per strand, train_keep.py; LOG [2]).

    bash run.sh train.py $DATA/runs/<name> --train $DATA/prep/train --val $DATA/prep/val [--mode flow|regress]
        [--epochs 300 --width 64 --lr 3e-4 --w_label 1 --w_disp 1]

Writes <run>/config.json (everything needed to rebuild the model + feature normalization), best.pt, last.pt,
train_log.jsonl. Loss is masked to the hair region (label channel) and to texels with kept strands
(displacement channels).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import features as F
import model as M


def load_dir(dirs):
    xs, ys, ms, ws = [], [], [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            xs.append(z["x"].astype(np.float32))
            ys.append(z["y"][1:])
            ms.append(z["region"])
            ws.append(z["w"] > 0)
    return [torch.from_numpy(np.stack(a)) for a in (xs, ys, ms, ws)]


def loss_fn(net, cfg, x, y, region, has, gen=None):
    B = x.shape[0]
    if cfg["mode"] == "flow":
        t = torch.rand(B, device=x.device)
        x0 = torch.randn_like(y)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * y
        pred = net(x, xt, t)
        err = (pred - (y - x0)) ** 2
    else:
        pred = net(x, torch.zeros_like(y), torch.zeros(B, device=x.device))
        err = (pred - y) ** 2
    hd = has.float()[:, None]
    l_disp = (err * hd).sum() / (hd.sum() * y.shape[1]).clamp(min=1)
    return l_disp, 0.0, l_disp.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--mode", default="flow", choices=["flow", "regress"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w_label", type=float, default=1.0)
    p.add_argument("--w_disp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feat_keep", action="store_true", help="the prep dirs hold features on the kept strands "
                   "(prep.py --keep_feat); at inference the features are then computed after the keep decision")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0, help="channel dropout on the input features")
    a = p.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    X, Y, Rg, H = load_dir(a.train)
    Xv, Yv, Rv, Hv = load_dir(a.val)
    # per-channel feature scale (over region texels); channel 0 (drawn label) and coordinates stay as they are
    sc = np.ones(X.shape[1], np.float32)
    for c in range(4, X.shape[1]):
        v = X[:, c][Rg].numpy()
        sc[c] = max(float(np.sqrt((v ** 2).mean())), 1e-3)
    cfg = dict(kind="disp", feat_keep=a.feat_keep, wd=a.wd, dropout=a.dropout, mode=a.mode, c_feat=int(X.shape[1]), c_out=int(Y.shape[1]), width=a.width, mults=[1, 2, 2, 4],
               feat_scale=sc.tolist(), feat_res=int(X.shape[-1]), w_label=a.w_label, w_disp=a.w_disp,
               knot_idx=list(F.KNOT_IDX), disp_scale=F.DISP_SCALE, train=a.train, val=a.val,
               n_train=int(len(X)), n_val=int(len(Xv)), epochs=a.epochs, lr=a.lr)
    json.dump(cfg, open(os.path.join(a.run_dir, "config.json"), "w"), indent=1)
    scale = torch.from_numpy(sc)[None, :, None, None]
    X, Xv = X / scale, Xv / scale
    fs = torch.from_numpy(F.flip_signs())[None, :, None, None]
    ts = torch.from_numpy(F.target_flip_signs()[1:])[None, :, None, None]
    net = M.build(cfg).to(dev)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, train {len(X)}, val {len(Xv)}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * max(1, len(X) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    best = 1e9
    logf = open(os.path.join(a.run_dir, "train_log.jsonl"), "w")
    g = torch.Generator(device=dev).manual_seed(123)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X))
        t0, tl = time.time(), []
        for i in range(0, len(X) - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            x, y, r, h = X[b].to(dev), Y[b].to(dev), Rg[b].to(dev), H[b].to(dev)
            flip = torch.rand(len(b), device=dev) < 0.5
            if flip.any():
                x[flip] = (x[flip] * fs.to(dev)).flip(-1)
                y[flip] = (y[flip] * ts.to(dev)).flip(-1)
                r[flip], h[flip] = r[flip].flip(-1), h[flip].flip(-1)
            if a.dropout > 0:
                x = x * (torch.rand(x.shape[0], x.shape[1], 1, 1, device=dev) > a.dropout).float()
            loss, _, _ = loss_fn(net, cfg, x, y, r, h)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                torch.manual_seed(1000)   # same t / noise every evaluation
                vl = [loss_fn(net, cfg, Xv[i:i + 32].to(dev), Yv[i:i + 32].to(dev), Rv[i:i + 32].to(dev),
                              Hv[i:i + 32].to(dev)) for i in range(0, len(Xv), 32)]
                # sample accuracy of the refined label vs the teacher's (thresholded), on changed / all texels
                xs = Xv.to(dev)
                pr = torch.cat([M.sample(net, xs[i:i + 32], cfg, 8) for i in range(0, len(xs), 32)])
                hv = Hv.to(dev)[:, None]
                nc = Yv.shape[1]
                d_err = (((pr - Yv.to(dev)) ** 2) * hv).sum().item() / max(hv.sum().item() * nc, 1)
                d_zero = ((Yv.to(dev) ** 2) * hv).sum().item() / max(hv.sum().item() * nc, 1)
            v = float(np.mean([x[0].item() for x in vl]))
            rec = dict(ep=ep, train=float(np.mean(tl)), val=v, disp_mse=d_err, disp_mse_zero=d_zero,
                       rel=d_err / max(d_zero, 1e-9), sec=round(time.time() - t0, 1))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
            torch.save(net.state_dict(), os.path.join(a.run_dir, "last.pt"))
            score = d_err / max(d_zero, 1e-9)   # sampled displacement error relative to "do not move"
            if score < best:
                best = score
                torch.save(net.state_dict(), os.path.join(a.run_dir, "best.pt"))


if __name__ == "__main__":
    main()
