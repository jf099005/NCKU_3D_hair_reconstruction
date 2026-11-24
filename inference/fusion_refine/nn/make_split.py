"""Split the hairstyles by base style (base_<k>_idx_*): variants of one base never cross splits.
Writes $DATA/split.json {"train": [...], "val": [...], "test": [...]}; stratified by the base's length class."""
import json
import os
import re
from collections import defaultdict

import numpy as np

import common

data = os.environ["DATA"]
lengths = json.load(open(os.path.join(data, "lengths.json")))
by_base = defaultdict(list)
for n in sorted(lengths):
    by_base[re.match(r"(base_\d+)_", n).group(1)].append(n)
rng = np.random.default_rng(0)
split = dict(train=[], val=[], test=[])
for cls in common.LENGTH_ORDER:
    bases = sorted(b for b, ns in by_base.items()
                   if common.length_class(np.median([lengths[n] for n in ns])) == cls)
    rng.shuffle(bases)
    k = len(bases)
    n_val, n_test = max(1, round(0.15 * k)), max(1, round(0.15 * k))
    for i, b in enumerate(bases):
        part = "val" if i < n_val else "test" if i < n_val + n_test else "train"
        split[part] += by_base[b]
for p, ns in split.items():
    cls = [common.length_class(lengths[n]) for n in ns]
    print(p, len(ns), {c: cls.count(c) for c in common.LENGTH_ORDER})
json.dump(split, open(os.path.join(data, "split.json"), "w"), indent=1)
