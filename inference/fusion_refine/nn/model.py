"""Conditional UNet on the scalp texture: (features, x_t, t) -> velocity (flow matching) or the target itself
(mode "regress"). Size / depth come from the checkpoint's config.json, not from this file."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


def t_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(tdim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(Fn.silu(self.n1(x)))
        s, b = self.film(te)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + s) + b
        h = self.c2(Fn.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, c_feat, c_out, width=64, mults=(1, 2, 2, 4), tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(c_feat + c_out, width, 3, padding=1)
        chs = [width * m for m in mults]
        self.down = nn.ModuleList()
        c = width
        skips = []
        for i, ch in enumerate(chs):
            self.down.append(nn.ModuleList([Block(c, ch, tdim), Block(ch, ch, tdim)]))
            skips.append(ch)
            c = ch
        self.mid = nn.ModuleList([Block(c, c, tdim), Block(c, c, tdim)])
        self.up = nn.ModuleList()
        for ch in reversed(chs):
            self.up.append(nn.ModuleList([Block(c + skips.pop(), ch, tdim), Block(ch, ch, tdim)]))
            c = ch
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c_out, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, feat, xt, t):
        te = self.tmlp(t_embed(t, self.tdim))
        h = self.inp(torch.cat([feat, xt], 1))
        hs = []
        for i, (b1, b2) in enumerate(self.down):
            h = b2(b1(h, te), te)
            hs.append(h)
            if i < len(self.down) - 1:
                h = Fn.avg_pool2d(h, 2)
        for b in self.mid:
            h = b(h, te)
        for i, (b1, b2) in enumerate(self.up):
            if i > 0:
                h = Fn.interpolate(h, scale_factor=2, mode="nearest")
            h = b2(b1(torch.cat([h, hs.pop()], 1), te), te)
        return self.out(h)


def build(cfg):
    return UNet(cfg["c_feat"], cfg["c_out"], cfg.get("width", 64), tuple(cfg.get("mults", (1, 2, 2, 4))))


@torch.no_grad()
def sample(net, feat, cfg, n_steps=16, generator=None):
    """feat (B, C, R, R) -> (B, c_out, R, R). flow: Euler from N(0, I) at t = 0 to the data at t = 1."""
    B, _, R, _ = feat.shape
    if cfg.get("mode", "flow") == "regress":
        return net(feat, torch.zeros(B, cfg["c_out"], R, R, device=feat.device), torch.zeros(B, device=feat.device))
    x = torch.randn(B, cfg["c_out"], R, R, device=feat.device, generator=generator)
    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=feat.device)
        x = x + net(feat, x, t) / n_steps
    return x
