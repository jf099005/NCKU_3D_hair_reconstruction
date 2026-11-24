"""Per-strand keep model: texture context CNN (scalp-texture features) sampled at each root + the strand's own
descriptors (features.strand_features) -> logit of FLIP (keep != drawn split). Sizes come from config.json."""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class KeepNet(nn.Module):
    def __init__(self, c_feat, d_strand, ctx=32, width=64, hidden=128):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.enc = nn.Sequential(blk(c_feat, width), blk(width, width), blk(width, width))
        self.down = nn.Sequential(nn.AvgPool2d(2), blk(width, width), blk(width, width))
        self.fuse = nn.Conv2d(2 * width, ctx, 1)
        self.mlp = nn.Sequential(nn.Linear(ctx + d_strand, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def context(self, feat):
        h = self.enc(feat)
        d = Fn.interpolate(self.down(h), size=h.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([h, d], 1))

    def forward(self, feat, sf, uv):
        """feat (B, C, R, R); sf (B, N, D); uv (B, N, 2) root uv in [0, 1] (row, col) -> (B, N) flip logits"""
        ctx = self.context(feat)
        grid = (uv[..., [1, 0]] * 2 - 1)[:, :, None]            # grid_sample wants (x = col, y = row)
        c = Fn.grid_sample(ctx, grid, mode="bilinear", align_corners=False)[..., 0].transpose(1, 2)
        return self.mlp(torch.cat([c, sf], -1))[..., 0]


def build(cfg):
    return KeepNet(cfg["c_feat"], cfg["d_strand"], cfg.get("ctx", 32), cfg.get("width", 64), cfg.get("hidden", 128))
