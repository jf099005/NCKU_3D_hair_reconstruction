"""Displacement by direct minimisation of a differentiable mixing objective (no teacher).

Per kept strand, a displacement at the knots KNOT_IDX (+ 0 at the root) is optimised with Adam:

    L = w_mix * mixing + w_mag * |D|^2 + w_rad * (D . n)^2 + w_smooth * |d2 D / dk2|^2
        + w_coh * |D - mean(D of K same-head neighbours)|^2 + w_len * (segment length change)^2

mixing = mean over the kept points of each head of the OTHER head's point density at that point; densities are
trilinear splats of the (moving) points into a voxel grid, blurred with a gaussian (both heads move, gradients
flow through both). Lengths are in units of SCALE (5 mm). The result goes through the same root ramp /
follow-the-leader / kink limit as features.apply (strand version: apply_knots).

Used directly per case (test-time optimisation) and as the training loss of an amortised model.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

import features as F

SCALE = 0.005            # m, unit of displacements in the loss
VOX = 0.003              # m
SIGMA_VOX = 1.0          # gaussian blur of the density, in voxels
STEP = 8                 # point subsampling (256 -> 32)

DEFAULTS = dict(iters=150, lr=0.3, w_mix=1.0, w_mag=0.02, w_rad=0.2, w_smooth=0.5, w_coh=0.5, w_len=5.0,
                k_nbr=8, d_max=0.02)


def interp_matrix(n_points=256, step=STEP, knots=F.KNOT_IDX):
    """(J, K) weights: displacement at subsampled point j = sum_k W[j, k] * D[k] (0 at the root, linear between
    knots), times the root ramp applied separately"""
    t = np.arange(0, n_points, step, dtype=float)
    xs = np.r_[0.0, np.array(knots, float)]
    W = np.zeros((len(t), len(knots)))
    for j, tt in enumerate(t):
        i = min(np.searchsorted(xs, tt, side="right") - 1, len(xs) - 2)
        w = (tt - xs[i]) / (xs[i + 1] - xs[i])
        if i > 0:
            W[j, i - 1] += 1 - w
        W[j, i] += w
    return W


def _gauss_kernel(sigma):
    r = int(np.ceil(2 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return torch.tensor(g / g.sum(), dtype=torch.float32)


def splat(P, lo, dims, vox):
    """P (N, 3) -> (1, 1, X, Y, Z) density, trilinear, differentiable w.r.t. P"""
    g = (P - lo) / vox
    i0 = torch.floor(g).long()
    f = g - i0
    i0 = i0.clamp(min=torch.zeros(3, dtype=torch.long, device=P.device), max=dims - 2)
    out = torch.zeros(int(dims[0] * dims[1] * dims[2]), device=P.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                idx = ((i0[:, 0] + dx) * dims[1] + (i0[:, 1] + dy)) * dims[2] + (i0[:, 2] + dz)
                out = out.index_add(0, idx, w)
    return out.view(1, 1, *[int(d) for d in dims])


def blur3(v, k):
    r = (len(k) - 1) // 2
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        v = Fn.conv3d(Fn.pad(v, pad), k.view(shape))
    return v


def sample(field, P, lo, dims, vox):
    """trilinear lookup of field (1, 1, X, Y, Z) at P (N, 3)"""
    g = (P - lo) / vox / (dims.float() - 1) * 2 - 1          # -> [-1, 1] per axis (x, y, z)
    grid = g[:, [2, 1, 0]].view(1, -1, 1, 1, 3)               # grid_sample: last dim = (z, y, x) -> (W, H, D)
    return Fn.grid_sample(field, grid, mode="bilinear", align_corners=True).view(-1)


def mixing_loss(PA, PB, lo, dims, vox, kern):
    """mean other-head density at the points of each head (densities normalised so that a uniformly filled
    voxel neighbourhood of the own head ~ 1)"""
    dA = blur3(splat(PA, lo, dims, vox), kern)
    dB = blur3(splat(PB, lo, dims, vox), kern)
    norm = 0.5 * (sample(dA, PA, lo, dims, vox).mean() + sample(dB, PB, lo, dims, vox).mean()).detach() + 1e-6
    return 0.5 * (sample(dB, PA, lo, dims, vox).mean() + sample(dA, PB, lo, dims, vox).mean()) / norm


def optimize(heads, keep, dev="cuda", verbose=False, **kw):
    """-> per head (S_kept, K, 3) world displacements at the knots (m), info.
    (img2hair_tsai_1 disables autograd globally at import, so gradients are switched on here explicitly.)"""
    with torch.enable_grad():
        return _optimize(heads, keep, dev, verbose, **kw)


def prepare(P_list, cfg=None, dev="cuda", n_points=256):
    """P_list: per head (S_kept, J, 3) subsampled points (m) of the kept strands -> state for objective()"""
    from scipy.spatial import cKDTree
    cfg = dict(DEFAULTS, **(cfg or {}))
    st = dict(cfg=cfg, W=torch.tensor(interp_matrix(n_points), dtype=torch.float32, device=dev),
              pts=[], ramps=[], nrm=[], nbrs=[], seg0=[])
    center = np.concatenate([P[:, 0] for P in P_list]).mean(0)
    for P in P_list:
        P = np.asarray(P, np.float64)
        s = np.concatenate([np.zeros((len(P), 1)), np.cumsum(np.linalg.norm(np.diff(P, axis=1), axis=-1), 1)], 1)
        x = np.clip(s / (0.15 * np.maximum(s[:, -1:], 1e-6)), 0, 1)
        st["ramps"].append(torch.tensor(x * x * (3 - 2 * x), dtype=torch.float32, device=dev))
        n = P - center
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        st["nrm"].append(torch.tensor(n, dtype=torch.float32, device=dev))
        k = min(cfg["k_nbr"] + 1, len(P))
        nb = cKDTree(P[:, 0]).query(P[:, 0], k=k)[1][:, 1:] if len(P) > 1 else np.zeros((len(P), 0), int)
        st["nbrs"].append(torch.tensor(nb, device=dev))
        Pt = torch.tensor(P / SCALE, dtype=torch.float32, device=dev)
        st["pts"].append(Pt)
        st["seg0"].append((Pt[:, 1:] - Pt[:, :-1]).norm(dim=-1))
    allP = np.concatenate([np.asarray(P).reshape(-1, 3) for P in P_list])
    vox = VOX / SCALE
    lo = allP.min(0) / SCALE - 4 * cfg["d_max"] / SCALE - 3 * vox
    hi = allP.max(0) / SCALE + 4 * cfg["d_max"] / SCALE + 3 * vox
    st.update(vox=vox, lo=torch.tensor(lo, dtype=torch.float32, device=dev),
              dims=torch.tensor(np.ceil((hi - lo) / vox).astype(int) + 2, device=dev),
              kern=_gauss_kernel(SIGMA_VOX).to(dev), dmax=cfg["d_max"] / SCALE)
    return st


def capped(d, st):
    return d * torch.clamp(st["dmax"] / d.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)


def moved_points(D, st):
    """D: per head (S, K, 3) knot displacements (units of SCALE) -> per head moved points (S, J, 3), capped D"""
    out, dds = [], []
    for d, p, r in zip(D, st["pts"], st["ramps"]):
        dd = capped(d, st)
        out.append(p + torch.einsum("jk,skc->sjc", st["W"], dd) * r[..., None])
        dds.append(dd)
    return out, dds


def objective(D, st):
    """-> loss, dict(mix, reg) for knot displacements D (per head (S, K, 3), units of SCALE)"""
    cfg = st["cfg"]
    Ps, dds = moved_points(D, st)
    reg = 0.0
    for dd, Q, p, n, nb, s0 in zip(dds, Ps, st["pts"], st["nrm"], st["nbrs"], st["seg0"]):
        if len(dd) == 0:
            continue
        disp = Q - p
        mag = (dd ** 2).sum(-1).mean()
        rad = ((disp * n).sum(-1) ** 2).mean()
        sm = ((dd[:, 2:] - 2 * dd[:, 1:-1] + dd[:, :-2]) ** 2).sum(-1).mean()
        coh = ((dd - dd[nb].mean(1)) ** 2).sum(-1).mean() if nb.shape[1] else 0.0
        ln = (((Q[:, 1:] - Q[:, :-1]).norm(dim=-1) - s0) ** 2).mean()
        reg = reg + cfg["w_mag"] * mag + cfg["w_rad"] * rad + cfg["w_smooth"] * sm + cfg["w_coh"] * coh \
            + cfg["w_len"] * ln
    mix = mixing_loss(Ps[0].reshape(-1, 3), Ps[1].reshape(-1, 3), st["lo"], st["dims"], st["vox"], st["kern"])
    return cfg["w_mix"] * mix + reg, dict(mix=mix, reg=reg)


def to_world(D, st):
    with torch.no_grad():
        return [capped(d, st).cpu().numpy().astype(np.float64) * SCALE for d in D]


def _optimize(heads, keep, dev="cuda", verbose=False, init=None, **kw):
    P_list = [h["positions"][kp][:, ::STEP] for h, kp in zip(heads, keep)]
    st = prepare(P_list, kw, dev, heads[0]["positions"].shape[1])
    cfg = st["cfg"]
    K = len(F.KNOT_IDX)
    D = [torch.zeros(len(p), K, 3, device=dev) if init is None else
         torch.tensor(i / SCALE, dtype=torch.float32, device=dev) for p, i in zip(st["pts"], init or [None] * 2)]
    D = [d.requires_grad_(True) for d in D]
    opt = torch.optim.Adam(D, lr=cfg["lr"])
    hist = []
    for it in range(cfg["iters"] + 1):
        loss, parts = objective(D, st)
        if it % 25 == 0 or it == cfg["iters"]:
            hist.append(dict(it=it, mix=float(parts["mix"]), reg=float(parts["reg"])))
            if verbose:
                print(hist[-1])
        if it == cfg["iters"]:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return to_world(D, st), dict(cfg=cfg, hist=hist)


def apply_knots(heads, keep, knots_world, max_kink=30.0):
    """knots_world: per head (S_kept, K, 3) displacements (m) -> edited heads, info (as features.apply)"""
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss
    out, info = [], {}
    for i, (h, kp, world) in enumerate(zip(heads, keep, knots_world)):
        idx = np.flatnonzero(kp)
        pos = h["positions"].copy()
        if idx.size:
            P0 = h["positions"][idx].astype(np.float64)
            full = F._interp_knots(world, np.array(F.KNOT_IDX, float), P0.shape[1])
            s, seg = dfm.arclength(P0)
            ramp = dfm.smoothstep(s / (0.15 * np.maximum(s[:, -1:], 1e-6)))
            moved = np.abs(full).max((1, 2)) > 2e-4
            full[~moved] = 0
            Q = P0 + ramp[..., None] * full
            mi = np.flatnonzero(moved)
            nk = 0
            if mi.size:
                Qi = dfm.follow_the_leader(Q[mi], seg[mi])
                Qi[:, 0] = P0[mi, 0]
                Q[mi] = Qi
                Q, nk = ss.limit_kinks(P0, Q, mi, max_kink)
            bad = ~np.isfinite(Q).all((1, 2))
            Q[bad] = P0[bad]
            dmm = np.linalg.norm(Q - P0, axis=-1).max(1) * 1e3
            tag = "base" if i == 0 else "donor"
            info.update({f"{tag}_moved_strands": int(moved.sum()), f"{tag}_kink_limited": int(nk),
                         f"{tag}_disp_mean_mm": float(dmm[moved].mean()) if moved.any() else 0.0,
                         f"{tag}_disp_p99_mm": float(np.percentile(dmm, 99))})
            pos[idx] = Q.astype(pos.dtype)
        out.append(dict(h, positions=pos))
    return out, info
