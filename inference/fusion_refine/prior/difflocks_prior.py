"""DiffLocks as a prior over hairstyles: 3D strands -> scalp texture (as data_processing/create_scalp_textures.py)
-> per-texel denoising error of the scalp-texture diffusion model (an ELBO-style "how far from the DiffLocks
distribution" map).

    prior = DiffLocksPrior()                         # loads checkpoints/ (run from the repo root)
    tex = prior.hair_to_texture(positions, root_uv)  # (65, 256, 256): 64 latent channels + density channel
    maps = prior.elbo_maps(tex, sigmas=(0.1, 0.3, 1.0), n_noise=4)   # {sigma: (256, 256) error map}

positions: (S, 256, 3) world, root_uv: (S, 2) in the scalp-texture layout ((row, col) / 256), the layout of the
benchmark heads and of DiffLocks' own output root_uv. Texture index = floor(root_uv * 256) as in the training
data (create_scalp_textures: [row = (1 - v) * res, col = u * res]).

Differences to the training data (relative comparisons only):
* density: this dataset copy has no density.png -> root count per texel, gaussian-smoothed, divided by its
  99th percentile over the hair region, clipped to [0, 1];
* hole filling: push-pull (the repo's CUDA extension) when it builds, else nearest-texel fill.
"""
import os

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CKPT_CODEC = os.path.join(REPO, "checkpoints", "strand_vae", "strand_codec.pt")
CKPT_DIFF = os.path.join(REPO, "checkpoints", "difflocks_diffusion", "scalp_v9_40k_06730000.pth")
CONFIG = os.path.join(REPO, "configs", "config_scalp_texture_conditional.json")
RES = 256


class DiffLocksPrior:
    def __init__(self, ckpt_codec=CKPT_CODEC, ckpt_diff=CKPT_DIFF, config=CONFIG, dev="cuda", random_init=False):
        """random_init: skip loading the checkpoints (code-path tests only; every number is meaningless)"""
        import k_diffusion as K
        from data_loader.dataloader import DEFAULT_BODY_DATA_DIR, DiffLocksDataset
        from models.strand_codec import StrandCodec
        self.dev = dev
        self.codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                                 nr_values_to_decode=255, dim_per_value_decoded=3).to(dev)
        if not random_init:
            self.codec.load_state_dict(torch.load(ckpt_codec, map_location=dev))
        self.codec.eval()
        cfg = K.config.load_config(config)
        self.model_config = cfg["model"]
        inner = K.config.make_model(cfg).to(dev).eval()
        self.denoiser = K.config.make_denoiser_wrapper(cfg)(inner).eval()
        if not random_init:
            ck = torch.load(ckpt_diff, map_location="cpu")
            self.denoiser.inner_model.load_state_dict(ck["model_ema"])
            del ck
        self.denoiser.to(dev)
        self.sigma_data = self.model_config["sigma_data"]
        self.norm = DiffLocksDataset.get_normalization_data(device=dev)
        self.scalp_trimesh, self.scalp = DiffLocksDataset.compute_scalp_data(os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply"))
        self._pushpull = None

    # ------------------------------------------------------------------ 3D -> latent texture
    def root_tbn(self, roots):
        """(S, 3) root positions -> (S, 3, 3) TBN (columns t, b, n) and normals, at the closest scalp point"""
        from data_loader.mesh_utils import closest_point_barycentrics, interpolate_tbn
        sc = self.scalp
        # igl wants int faces and matching float types (the mesh data keeps faces unsigned)
        V = np.ascontiguousarray(sc["verts"], np.float64)
        _, barys, vidx, _ = closest_point_barycentrics(np.ascontiguousarray(roots, np.float64), V,
                                                       np.ascontiguousarray(sc["faces"], np.int64))
        t, b, n = interpolate_tbn(barys, vidx, sc["v_tangents"], sc["v_bitangents"], sc["v_normals"])
        t, b, n = [x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x) for x in (t, b, n)]
        return np.stack((t, b, n), axis=2).astype(np.float32), n.astype(np.float32)

    def encode_torch(self, P, tbn, nrm):
        """differentiable: P (S, 256, 3) world torch, tbn / nrm torch (constant) -> (S, 64)"""
        from data_loader.mesh_utils import world_to_tbn_space
        from models.strand_codec import normalize_gt_data
        from utils.strand_util import compute_dirs
        loc, _ = world_to_tbn_space(tbn[None].clone(), P[None], nrm[None])
        loc = loc.reshape(-1, 256, 3)
        g = normalize_gt_data(dict(strand_positions=loc, strand_directions=compute_dirs(loc, append_last_dir=False)),
                              self.norm)
        return self.codec.encoder(g)["z"].float()

    @torch.no_grad()
    def encode(self, positions, chunk=8000):
        """(S, 256, 3) world positions -> (S, 64) strand latents (TBN at the closest scalp point, as the dataset)"""
        P = np.asarray(positions, np.float32)
        tbn, n = self.root_tbn(P[:, 0])
        out = []
        for i in range(0, len(P), chunk):
            out.append(self.encode_torch(torch.tensor(P[i:i + chunk], device=self.dev),
                                         torch.tensor(tbn[i:i + chunk], device=self.dev),
                                         torch.tensor(n[i:i + chunk], device=self.dev)))
        return torch.cat(out)

    def _fill(self, tex, mask, method="push_pull"):
        """fill texels without strands: push-pull extension (as the training data) if it builds, else / with
        method="nearest" the nearest texel (= what texture_builder does, differentiable)"""
        if method == "push_pull" and self._pushpull is None:
            try:
                from torch.utils.cpp_extension import load
                self._pushpull = load(name="push_pull_inpaint_cpp",
                                      sources=[os.path.join(REPO, "extensions", "push_pull_inpaint.cpp"),
                                               os.path.join(REPO, "extensions", "cuda", "push_pull_inpaint.cu")],
                                      extra_include_paths=[os.path.join(REPO, "extensions", "include")])
            except Exception as e:   # no nvcc / build failure
                print("push-pull extension unavailable, nearest fill:", str(e).splitlines()[0][:120])
                self._pushpull = False
        if method == "push_pull" and self._pushpull:
            return self._pushpull.push_pull_inpaint(mask[None].float(), tex[None])[0], "push_pull"
        from scipy.ndimage import distance_transform_edt
        _, (ir, ic) = distance_transform_edt(~mask.cpu().numpy(), return_indices=True)
        return tex[:, torch.from_numpy(ir).to(tex.device), torch.from_numpy(ic).to(tex.device)], "nearest"

    @torch.no_grad()
    def hair_to_texture(self, positions, root_uv, latents=None, fill="push_pull"):
        """-> (65, RES, RES) diffusion input (64 latent channels + density), info"""
        from scipy.ndimage import gaussian_filter
        z = self.encode(positions) if latents is None else latents
        rc = np.clip(np.floor(np.asarray(root_uv) * RES).astype(np.int64), 0, RES - 1)
        r, c = torch.from_numpy(rc[:, 0]).to(self.dev), torch.from_numpy(rc[:, 1]).to(self.dev)
        tex = torch.zeros(64, RES, RES, device=self.dev)
        cnt = torch.zeros(RES, RES, device=self.dev)
        idx = r * RES + c
        tex.view(64, -1).index_add_(1, idx, z.T)
        cnt.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        has = cnt > 0
        tex[:, has] /= cnt[has]
        tex, fill = self._fill(tex, has, fill)
        d = gaussian_filter(cnt.cpu().numpy(), 2.0)
        q = np.percentile(d[has.cpu().numpy()], 99) if has.any() else 1.0
        d = np.clip(d / max(q, 1e-6), 0, 1)
        dens = (torch.tensor(d, device=self.dev, dtype=torch.float32) - 0.5) / (0.5 / self.sigma_data)
        return torch.cat([tex, dens[None]]), dict(fill=fill, texels=int(has.sum()), strands=len(rc), has=has.cpu().numpy())

    # ------------------------------------------------------------------ ELBO-style maps
    @torch.no_grad()
    def elbo_maps(self, tex, sigmas=(0.05, 0.1, 0.3, 1.0), n_noise=4, seed=0, batch=4):
        """per sigma: (RES, RES) mean squared denoising error over the 64 latent channels, / sigma_data^2,
        unconditional branch, same noise draws for every texture (seeded) so maps of different textures compare"""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        x = tex[None].float()
        out = {}
        for s in sigmas:
            err = torch.zeros(RES, RES, device=self.dev)
            for i in range(0, n_noise, batch):
                k = min(batch, n_noise - i)
                eps = torch.randn((k,) + tex.shape, device=self.dev, generator=g)
                xn = x + s * eps
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    den = self.denoiser(xn, torch.full((k,), s, device=self.dev))
                err += ((den.float() - x)[:, :64] ** 2).mean(1).sum(0)
            out[float(s)] = (err / n_noise / self.sigma_data ** 2).cpu().numpy()
        return out


    # ------------------------------------------------------------------ (a) exact log-likelihood, per texel
    def loglik_map(self, tex, n_steps=40, sigma_min=None, sigma_max=None, seed=0, latent_only=True):
        """log p(x) of the texture under the probability-flow ODE (unconditional), with Hutchinson's divergence
        estimate split per element (E[v_i (J^T v)_i] = J_ii), so the per-texel sum of the maps is the estimate of
        log p. Heun steps on a log-spaced sigma grid sigma_min -> sigma_max (k_diffusion.sampling.log_likelihood
        does the same with an adaptive dopri5 and only returns the total).
        -> dict(total, bits_per_dim, map (RES, RES): prior + divergence per texel, div_map, prior_map, fevals)
        latent_only: maps sum the 64 latent channels only (the total always uses all 65)."""
        mc = self.model_config
        smin = sigma_min or mc["sigma_min"]
        smax = sigma_max or mc["sigma_max"]
        sig = torch.exp(torch.linspace(np.log(smin), np.log(smax), n_steps + 1, device=self.dev))
        g = torch.Generator(device=self.dev).manual_seed(seed)
        v = (torch.randint(0, 2, (1,) + tex.shape, device=self.dev, generator=g) * 2 - 1).float()
        fe = [0]

        def f(x, s):
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    den = self.denoiser(x, s.reshape(1))
                d = (x - den.float()) / s
                gr = torch.autograd.grad((d * v).sum(), x)[0]
            fe[0] += 1
            return d.detach(), (v * gr).detach()          # drift, per-element divergence estimate

        x = tex[None].float()
        div = torch.zeros_like(x)
        for i in range(n_steps):
            s0, s1 = sig[i], sig[i + 1]
            h = s1 - s0
            d0, dv0 = f(x, s0)
            xe = x + h * d0
            d1, dv1 = f(xe, s1)
            x = x + h * (d0 + d1) / 2
            div = div + h * (dv0 + dv1) / 2
        prior = torch.distributions.Normal(0, smax).log_prob(x)
        tot = float((prior + div).sum())
        ch = slice(0, 64) if latent_only else slice(None)
        N = tex.numel()
        return dict(total=tot, bits_per_dim=-tot / (N * np.log(2)),
                    map=(prior + div)[0, ch].sum(0).cpu().numpy(), div_map=div[0, ch].sum(0).cpu().numpy(),
                    prior_map=prior[0, ch].sum(0).cpu().numpy(), fevals=fe[0])

    # ------------------------------------------------------------------ (c) score: differentiable texture, SDS, SDEdit
    def texture_builder(self, roots, root_uv):
        """constant parts of the strand -> texture map for strands with fixed roots (TBN, texel index, hole-filling
        indices, density). Returns build(P) -> (65, RES, RES), differentiable w.r.t. P (S, 256, 3) torch."""
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        tbn, n = self.root_tbn(roots)
        tbn_t, n_t = torch.tensor(tbn, device=self.dev), torch.tensor(n, device=self.dev)
        rc = np.clip(np.floor(np.asarray(root_uv) * RES).astype(np.int64), 0, RES - 1)
        idx = torch.from_numpy(rc[:, 0] * RES + rc[:, 1]).to(self.dev)
        cnt = np.zeros(RES * RES)
        np.add.at(cnt, rc[:, 0] * RES + rc[:, 1], 1)
        cnt = cnt.reshape(RES, RES)
        has = cnt > 0
        _, (ir, ic) = distance_transform_edt(~has, return_indices=True)
        fill = torch.from_numpy((ir * RES + ic).reshape(-1)).to(self.dev)
        cnt_t = torch.tensor(np.maximum(cnt, 1).reshape(-1), device=self.dev, dtype=torch.float32)
        d = gaussian_filter(cnt, 2.0)
        d = np.clip(d / max(np.percentile(d[has], 99), 1e-6), 0, 1)
        dens = ((torch.tensor(d, device=self.dev, dtype=torch.float32) - 0.5) / (0.5 / self.sigma_data))[None]

        def build(P, chunk=8000):
            z = torch.cat([self.encode_torch(P[i:i + chunk], tbn_t[i:i + chunk], n_t[i:i + chunk])
                           for i in range(0, len(P), chunk)])
            t = torch.zeros(64, RES * RES, device=self.dev).index_add(1, idx, z.T) / cnt_t
            return torch.cat([t[:, fill].view(64, RES, RES), dens])
        build.has = has
        return build

    def sds_loss(self, tex, mask=None, sigma_range=(0.05, 1.0), n=1, generator=None):
        """Score-distillation surrogate: its gradient w.r.t. tex is w(sigma) * (tex - D(tex + sigma eps, sigma)) on the
        64 latent channels (mask: (RES, RES) bool / float weights, e.g. the boundary band), sigma log-uniform.
        w = 1 / sigma_data^2 here (x0-space SDS); the denoiser call is not differentiated."""
        lo, hi = np.log(sigma_range[0]), np.log(sigma_range[1])
        loss = 0.0
        m = torch.ones(RES, RES, device=self.dev) if mask is None else torch.as_tensor(mask, device=self.dev).float()
        for _ in range(n):
            s = float(np.exp(lo + (hi - lo) * torch.rand(1, generator=generator).item()))
            with torch.no_grad():
                eps = torch.randn((1,) + tex.shape, device=self.dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    den = self.denoiser(tex.detach()[None] + s * eps, torch.full((1,), s, device=self.dev)).float()[0]
                grad = (tex.detach() - den)[:64] / self.sigma_data ** 2 * m
            loss = loss + (grad * tex[:64]).sum() / m.sum().clamp(min=1)
        return loss / n

    @torch.no_grad()
    def sdedit(self, tex, band, sigma_start=0.5, n_steps=20, seed=0, rho=7.0):
        """Re-denoise the texture from sigma_start inside `band` ((RES, RES) bool), keeping the rest (RePaint:
        after every step the texels outside the band are replaced by tex + sigma_next * eps). Heun steps on the
        Karras schedule sigma_start -> sigma_min, unconditional. -> new (65, RES, RES)"""
        import k_diffusion as K
        g = torch.Generator(device=self.dev).manual_seed(seed)
        sig = K.sampling.get_sigmas_karras(n_steps, self.model_config["sigma_min"], sigma_start, rho=rho, device=self.dev)
        keep = torch.as_tensor(~np.asarray(band, bool), device=self.dev)
        x0 = tex[None].float()
        x = x0 + sig[0] * torch.randn(x0.shape, device=self.dev, generator=g)

        def D(x, s):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self.denoiser(x, s.reshape(1)).float()
        for i in range(len(sig) - 1):
            s0, s1 = sig[i], sig[i + 1]
            d0 = (x - D(x, s0)) / s0
            xn = x + (s1 - s0) * d0
            if s1 > 0:
                d1 = (xn - D(xn, s1)) / s1
                xn = x + (s1 - s0) * (d0 + d1) / 2
            known = x0 + s1 * torch.randn(x0.shape, device=self.dev, generator=g)
            x = torch.where(keep, known, xn)
        return x[0]

    @torch.no_grad()
    def decode(self, latents, root_uv, roots=None, chunk=8000):
        """(S, 64) latents at root_uv (texture layout) -> (S, 256, 3) world strands (DiffLocks decoder +
        tbn_space_to_world); with roots, each strand is translated so its root is exactly there"""
        from data_loader.mesh_utils import tbn_space_to_world
        pts = torch.cat([self.codec.decoder(latents[i:i + chunk], None, self.norm)["strand_positions"]
                         for i in range(0, len(latents), chunk)])
        w = tbn_space_to_world(torch.as_tensor(np.asarray(root_uv), dtype=torch.float32), pts, self.scalp)
        w = w.detach().cpu().numpy() if torch.is_tensor(w) else np.asarray(w)
        if roots is not None:
            w = w - w[:, :1] + np.asarray(roots)[:, None]
        return w
