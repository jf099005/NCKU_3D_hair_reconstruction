"""benchmark/build_benchmark.py with hair lengths from full_strands.npz (this dataset copy has no
guide_strands.npz). Lengths are cached in $DATA/lengths.json. Same arguments as build_benchmark.py."""
import json
import os
import sys

import numpy as np

import common
import build_benchmark

CACHE = os.path.join(os.environ.get("DATA", "."), "lengths.json")


def _lengths():
    if os.path.isfile(CACHE):
        return json.load(open(CACHE))
    return {}


def hairstyle_length(dataset_path, name, _c=_lengths()):
    if name not in _c:
        p = np.load(os.path.join(common.hairstyle_dir(dataset_path, name), "full_strands.npz"))["positions"]
        sub = p[:: max(1, len(p) // 2000)]
        _c[name] = float(np.median(np.linalg.norm(np.diff(sub, axis=1), axis=-1).sum(1)))
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(_c, open(CACHE, "w"), indent=1)
    return _c[name]


common.hairstyle_length = hairstyle_length

# --names_from <split.json>:<part>: sample only hairstyles of that split part
if "--names_from" in sys.argv:
    i = sys.argv.index("--names_from")
    f, part = sys.argv[i + 1].rsplit(":", 1)
    del sys.argv[i:i + 2]
    allowed = set(json.load(open(f))[part])
    _orig_list = common.list_hairstyles
    common.list_hairstyles = lambda *a, **k: [n for n in _orig_list(*a, **k) if n in allowed]
if "--dataset_path" not in sys.argv:
    sys.argv += ["--dataset_path", os.environ["DATASET"]]
build_benchmark.main()
