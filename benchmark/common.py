"""Shared helpers of the fusion benchmark: paths, dataset hairstyles, label maps, json io.

Layout convention: every mask / label map of the benchmark is in the DiffLocks **scalp-texture
layout** (row, col of the res x res texture, res = 256), the layout the fusion masks of
inference/img2hair_tsai_1.py use. Dataset hairstyles are converted to it on load (load_hairstyle),
so their `root_uv` means the same as DiffLocks' output `root_uv` = (row, col) / res and
img2hair_tsai_1.root_texels works on both.
"""
import datetime
import json
import os
import sys

import cv2
import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCH_DIR, ".."))
# P76154862/ (holds NCKU_3D_hair_reconstruction/, highlighting/ and DiffLocks_Dataset/)
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
DEFAULT_DATASET_PATH = os.path.join(WORKSPACE_ROOT, "DiffLocks_Dataset", "difflocks_dataset")
DEFAULT_STRANDS_FILE = "full_strands.npz"
TEXTURE_RES = 256           # DiffLocks scalp texture resolution (config input_size)
BLENDER_PATH = "/home/kyh/blender/blender"

for _p in (REPO_ROOT, WORKSPACE_ROOT, os.path.join(REPO_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------- dataset hairstyles

def list_hairstyles(dataset_path, strands_file=DEFAULT_STRANDS_FILE):
    """Names of the hairstyles under <dataset_path>/generated_hairstyles that have `strands_file`
    and metadata.json (the dataset may be only partly extracted)."""
    root = os.path.join(dataset_path, "generated_hairstyles")
    return sorted(name for name in os.listdir(root)
                  if os.path.isfile(os.path.join(root, name, strands_file))
                  and os.path.isfile(os.path.join(root, name, "metadata.json")))


def hairstyle_dir(dataset_path, name):
    return os.path.join(dataset_path, "generated_hairstyles", name)


def hairstyle_length(dataset_path, name):
    """median strand length (m) of a hairstyle, from its small guide_strands.npz"""
    p = np.load(os.path.join(hairstyle_dir(dataset_path, name), "guide_strands.npz"))["positions"]
    return float(np.median(np.linalg.norm(np.diff(p, axis=1), axis=-1).sum(1)))


# hair length classes by median strand length (m; the head is ~0.16 m wide):
#   short  < 5 cm    buzz / crew cut / pixie: the hair lies on the scalp
#   mid    5-15 cm   covers the ears, down to about the chin (bob)
#   long   >= 15 cm  past the chin, shoulder length and longer
LENGTH_CLASSES = (("short", 0.0, 0.05), ("mid", 0.05, 0.15), ("long", 0.15, float("inf")))
LENGTH_ORDER = {name: i for i, (name, _, _) in enumerate(LENGTH_CLASSES)}


def length_class(median_length):
    for name, lo, hi in LENGTH_CLASSES:
        if lo <= median_length < hi:
            return name
    raise ValueError(f"bad hair length {median_length}")


def length_pair(classes):
    """unordered combination of the heads' length classes, e.g. "short+long" (any m)"""
    return "+".join(sorted(classes, key=LENGTH_ORDER.__getitem__))


def all_length_pairs(m=2):
    import itertools
    return [length_pair(c) for c in itertools.combinations_with_replacement(LENGTH_ORDER, m)]


def length_standard():
    """the class table as JSON (benchmark.json / report)"""
    return [dict(name=n, min_m=lo, max_m=None if hi == float("inf") else hi) for n, lo, hi in LENGTH_CLASSES]


def gt_root_uv_to_texture_uv(root_uv, mirror_hair):
    """Dataset root_uv (u, v) -> scalp-texture uv (row, col) / res.

    Same mapping as data_processing/create_scalp_textures.py, which built the textures DiffLocks
    is trained on: the dataloader first flips u of mirrored hairstyles (the Blender pipeline
    mirrored the strands but not their uvs), then texel = (row, col) = ((1 - v) * res, u * res)."""
    uv = np.asarray(root_uv, dtype=np.float32)
    u = 1.0 - uv[:, 0] if mirror_hair else uv[:, 0]
    return np.stack([1.0 - uv[:, 1], u], axis=1).astype(np.float32)


def load_hairstyle(npz_path, max_strands=None, seed=0, root_uv_only=False):
    """One dataset hairstyle as {"positions" (S, P, 3), "root_uv" (S, 2) in the scalp-texture
    layout, "strand_idx" (S,) indices into the npz}; with `max_strands`, a random subset
    (fixed by `seed`) of that many strands. root_uv_only skips the (large) positions."""
    meta_path = os.path.join(os.path.dirname(npz_path), "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        mirror = bool(json.load(f).get("mirror_hair", False))
    d = np.load(npz_path)
    root_uv = gt_root_uv_to_texture_uv(d["root_uv"], mirror)
    idx = np.arange(root_uv.shape[0])
    if max_strands is not None and max_strands < idx.size:
        idx = np.sort(np.random.default_rng(seed).choice(idx.size, max_strands, replace=False))
    head = {"root_uv": root_uv[idx], "strand_idx": idx, "mirror_hair": mirror, "npz_path": npz_path}
    if not root_uv_only:
        head["positions"] = np.ascontiguousarray(d["positions"][idx], dtype=np.float32)
    return head


def root_texels(root_uv, res=TEXTURE_RES):
    """(row_idx, col_idx) texel of every root (same as img2hair_tsai_1.root_texels)."""
    rc = np.clip(np.floor(np.asarray(root_uv) * res).astype(np.int64), 0, res - 1)
    return rc[:, 0], rc[:, 1]


def hair_region(root_uvs, res=TEXTURE_RES, close_px=5):
    """(res, res) bool: texels that carry roots of any of the given heads, holes closed - the part
    of the scalp texture where a fusion mask actually decides something."""
    occ = np.zeros((res, res), np.uint8)
    for uv in root_uvs:
        occ[root_texels(uv, res)] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    return cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k).astype(bool)


# ---------------------------------------------------------------- label maps <-> masks
# A fusion of m heads is described by an (res, res) int label map: 0 = head 0 (base), k = head k.
# On disk (and for the fusion scripts) it is m-1 binary masks: mask_k.png = (label == k), white =
# head k's region, the same convention as mask_file_paths of the fusion configs.

def labels_to_masks(labels, m):
    return [labels == k for k in range(1, m)]


def masks_to_labels(masks):
    """Inverse of labels_to_masks; where masks overlap the later one wins."""
    labels = np.zeros(masks[0].shape, np.int16)
    for k, mk in enumerate(masks, start=1):
        labels[np.asarray(mk, bool)] = k
    return labels


def save_mask_png(mask, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.asarray(mask, bool).astype(np.uint8) * 255)


def load_mask_png(path, res=TEXTURE_RES):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != (res, res):
        m = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
    return m > 127


# categorical colors per head index (base blue, donor orange, then more for m > 2)
HEAD_COLORS = ["#2a78d6", "#eb6834", "#2fa36b", "#9b59b6", "#d4a017", "#17a2b8"]


def _hex_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def save_label_preview(labels, region, path):
    """Label map as a color png (head colors, outside the hair region darkened)."""
    img = np.zeros(labels.shape + (3,), np.uint8)
    for k in range(int(labels.max()) + 1):
        img[labels == k] = _hex_bgr(HEAD_COLORS[k % len(HEAD_COLORS)])
    if region is not None:
        img[~region] = (img[~region] * 0.3).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------- json io

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
