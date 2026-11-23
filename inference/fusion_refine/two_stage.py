"""Helpers of the two-stage (coarse strand subset -> fine band) joint FM (benchmark/LOG.md [16]-[18]):
the coarse stage runs on a strand subset and a coarse mask (block majority), its changed cells are upsampled,
and the fine stage refines every strand inside the band around them (adaptive.refine_joint_fm with node_filter,
edge_rows and lam_scope="movable")."""
import numpy as np

def block_majority(labels, res):
    k = labels.shape[0] // res
    return (labels.reshape(res, k, res, k).mean((1, 3)) > 0.5).astype(labels.dtype) if k > 1 else labels


def upsample(a, k):
    return np.kron(a, np.ones((k, k), a.dtype)) if k > 1 else a


def boundary(mask, region):
    b = np.zeros_like(mask)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        b |= region & np.roll(region, (dr, dc), (0, 1)) & (np.roll(mask, (dr, dc), (0, 1)) != mask)
    return b
