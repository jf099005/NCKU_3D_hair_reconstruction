"""Small shared helpers (scalp-texture layout)."""
import numpy as np

TEXTURE_RES = 256   # DiffLocks scalp texture resolution


def root_texels(root_uv, res=TEXTURE_RES):
    """(row_idx, col_idx) texel of every root (same as img2hair_tsai_1.root_texels / benchmark common)."""
    rc = np.clip(np.floor(np.asarray(root_uv) * res).astype(np.int64), 0, res - 1)
    return rc[:, 0], rc[:, 1]


def hair_region(root_uvs, res=TEXTURE_RES, close_px=5):
    """(res, res) bool: texels carrying roots of any head, holes closed (same as benchmark common.hair_region)."""
    import cv2
    occ = np.zeros((res, res), np.uint8)
    for uv in root_uvs:
        occ[root_texels(uv, res)] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    return cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k).astype(bool)


def initial_keep(heads, labels):
    """per head: the strands whose root texel the label map gives to that head"""
    res = labels.shape[0]
    return [labels[root_texels(h["root_uv"], res)] == i for i, h in enumerate(heads)]


def keep_from_mask(heads, mask):
    """per head: strands whose root texel the (res, res) bool mask (True = head 1) gives to that head"""
    res = mask.shape[0]
    return [mask[root_texels(h["root_uv"], res)] == bool(i) for i, h in enumerate(heads)]


def boundary(mask, region):
    """texels of `region` with a 4-neighbour in `region` of the other label"""
    b = np.zeros_like(mask)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        b |= region & np.roll(region, (dr, dc), (0, 1)) & (np.roll(mask, (dr, dc), (0, 1)) != mask)
    return b
