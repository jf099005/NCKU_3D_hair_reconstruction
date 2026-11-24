"""Amortised displacement model: predicts the knot displacements that diffopt.optimize would find, trained by
minimising diffopt.objective directly (no teacher, no labels).

Per kept strand the input is
    per point (J subsampled points): own / other head density at the point (log1p, from the blurred splats of
    the current points), the other head's density gradient (finite differences), outward normal, point - root
    + the strand descriptors (features.strand_features, normalised with the run's statistics).
An MLP encodes each strand, one neighbour-averaging step (K same-head neighbours by root) mixes the codes, and a
head outputs (K_knots, 3) displacement updates in units of diffopt.SCALE. With n_pass > 1 the model is applied
again on the moved points (fields recomputed), the updates add up. Sizes come from the run's config.json.
"""
import numpy as np
import torch
import torch.nn as nn

import diffopt as DO
import features as F

N_PT_FEAT = 11


def point_features(st, Ps):
    """Ps: per head current points (S, J, 3) in SCALE units -> per head (S, J, N_PT_FEAT), no grad"""
    with torch.no_grad():
        lo, dims, vox, kern = st["lo"], st["dims"], st["vox"], st["kern"]
        dens = [DO.blur3(DO.splat(P.reshape(-1, 3), lo, dims, vox), kern) for P in Ps]
        norm = [max(float(DO.sample(d, P.reshape(-1, 3), lo, dims, vox).mean()), 1e-6) for d, P in zip(dens, Ps)]
        out = []
        for h, P in enumerate(Ps):
            S, J, _ = P.shape
            q = P.reshape(-1, 3)
            own = DO.sample(dens[h], q, lo, dims, vox) / norm[h]
            oth_f = dens[1 - h]
            oth = DO.sample(oth_f, q, lo, dims, vox) / norm[h]
            e = vox
            g = torch.stack([(DO.sample(oth_f, q + e * u, lo, dims, vox) - DO.sample(oth_f, q - e * u, lo, dims, vox))
                             / (2 * e) for u in torch.eye(3, device=q.device)], -1) / norm[h]
            f = torch.cat([torch.log1p(own)[:, None], torch.log1p(oth)[:, None], torch.tanh(g),
                           st["nrm"][h].reshape(-1, 3), (P - P[:, :1]).reshape(-1, 3) / 20.0], -1)
            out.append(f.view(S, J, N_PT_FEAT))
        return out


class DispNet(nn.Module):
    def __init__(self, n_pt, d_strand, n_knots, hidden=256, n_pass=2):
        super().__init__()
        self.n_pass = n_pass
        self.enc = nn.Sequential(nn.Linear(n_pt * N_PT_FEAT + d_strand + n_knots * 3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.mix = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, n_knots * 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.n_knots = n_knots

    def forward(self, st, sf):
        """st: diffopt.prepare state; sf: per head (S, d_strand) normalised descriptors -> per head (S, K, 3)"""
        D = [torch.zeros(len(p), self.n_knots, 3, device=p.device) for p in st["pts"]]
        for _ in range(self.n_pass):
            Ps, _ = DO.moved_points(D, st)
            pf = point_features(st, [P.detach() for P in Ps])
            newD = []
            for h in range(len(D)):
                if len(D[h]) == 0:
                    newD.append(D[h])
                    continue
                x = torch.cat([pf[h].flatten(1), sf[h], D[h].flatten(1)], -1)
                z = self.enc(x)
                nb = st["nbrs"][h]
                zn = z[nb].mean(1) if nb.shape[1] else z
                z = self.mix(torch.cat([z, zn], -1))
                newD.append(D[h] + self.out(z).view(-1, self.n_knots, 3))
            D = newD
        return D


def build(cfg):
    return DispNet(cfg["n_pt"], cfg["d_strand"], cfg["n_knots"], cfg.get("hidden", 256), cfg.get("n_pass", 2))


def case_inputs(heads, keep, labels, cfg, dev="cuda"):
    """state + normalised strand descriptors of one case (keep = the kept strands to move)"""
    fe = F.features(heads, labels, keep=keep)
    sf = F.strand_features(heads, labels, fe, keep=keep)
    st = DO.prepare([h["positions"][k][:, ::DO.STEP] for h, k in zip(heads, keep)], cfg.get("obj"), dev,
                    heads[0]["positions"].shape[1])
    mu, sc = np.array(cfg["sf_mean"], np.float32), np.array(cfg["sf_scale"], np.float32)
    sft = [torch.tensor((s[k] - mu) / sc, dtype=torch.float32, device=dev) for s, k in zip(sf, keep)]
    return st, sft
