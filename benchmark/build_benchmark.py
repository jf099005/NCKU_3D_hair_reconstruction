#!/usr/bin/env python3
"""Build a hair-fusion benchmark: n test cases, each = m dataset hairstyles + m - 1 fusion masks.

    python benchmark/build_benchmark.py N [--m 2] [--out benchmark/data/<name>] [--seed 0]
           [--n_short 20 --n_mid 20 --n_long 20] [--min_hair_length 0.0 --max_hair_length inf]
           [--mask_generators left_right,front_back,half_plane,polygon] [--min_frac 0.15]

The cases sample their hairstyles from a pool: the dataset hairstyles inside [min, max]_hair_length, of which
--n_short / --n_mid / --n_long hairstyles of each length class (common.LENGTH_CLASSES) are drawn at random
(default: all of that class; 0 = leave the class out). Hairstyles are NOT copied: each case stores the path
of the dataset's strands npz. Output:

    <out>/benchmark.json              settings, pool (hairstyles + their lengths / classes), list of case ids
    <out>/length_distribution.png     median strand length of the available hairstyles vs the pool, pool
                                      size per length class, cases per length pair, case ratio
                                      per length pair as a class x class matrix (vs expected)
    <out>/cases/<case_id>/case.json   heads (name + npz path + median length + length class, head 0 = base),
                                      length_pair (e.g. "short+long", common.LENGTH_CLASSES), masks, seed
    <out>/cases/<case_id>/mask_<k>.png   head k's region (white), scalp-texture layout, k = 1 .. m-1
    <out>/cases/<case_id>/labels.png     preview: label map in head colors, outside the hair region dark
"""
import argparse
import os

import numpy as np

import common
from masks import MASK_GENERATORS, generate_labels

BENCHMARK_VERSION = 2

# reference categorical palette slots 1-3 (validated as a set): one color per length class
CLASS_COLORS = {"short": "#2a78d6", "mid": "#eb6834", "long": "#1baf7a"}
INK, INK_2, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
AVAILABLE_COLOR = "#d6d5cf"


def select_pool(names, lengths, counts, seed):
    """per length class, `counts[class]` random hairstyles of that class from `names` (None = all of them).
    Its own rng (from the same seed), so the pool draw is reproducible and separate from the cases' stream."""
    rng = np.random.default_rng([seed, 1])
    pool = []
    for cls in common.LENGTH_ORDER:
        members = [n for n in names if common.length_class(lengths[n]) == cls]
        k = counts.get(cls)
        if k is None or k >= len(members):
            pool += members
        else:
            pool += [members[i] for i in sorted(rng.choice(len(members), k, replace=False))]
    return sorted(pool)


def expected_pair_ratios(pool, lengths):
    """share of each 2-hair length pair when a case draws 2 distinct hairstyles uniformly from the pool:
    C(n_a, 2) / C(N, 2) for a + a, n_a * n_b / C(N, 2) for a + b"""
    n = {c: sum(common.length_class(lengths[h]) == c for h in pool) for c in common.LENGTH_ORDER}
    total = len(pool) * (len(pool) - 1) / 2
    out = {}
    for p in common.all_length_pairs(2):
        a, b = p.split("+")
        out[p] = (n[a] * (n[a] - 1) / 2 if a == b else n[a] * n[b]) / total if total else 0.0
    return out


def plot_length_distribution(path, lengths, available, pool, cases, title):
    """four panels: histogram of median strand length (available vs pool, pool colored by class, class
    thresholds), pool size per class (vs available), number of cases per length pair, and (m = 2) the
    class x class matrix of the case ratio per length pair next to the ratio expected from the pool"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(20.5, 4.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=[2.3, 1, 1.4, 1.45], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # (1) histogram, 1 cm bins; available in recessive gray behind, pool on top colored by class
    cm = {n: lengths[n] * 100 for n in available}
    top = max(cm.values())
    bins = np.arange(0, np.ceil(top) + 1.01, 1.0)
    ax = axes[0]
    ax.hist([cm[n] for n in available], bins=bins, color=AVAILABLE_COLOR, rwidth=0.88)
    pool_by_cls = [[cm[n] for n in pool if common.length_class(lengths[n]) == c] for c in common.LENGTH_ORDER]
    ax.hist(pool_by_cls, bins=bins, stacked=True, rwidth=0.88, color=[CLASS_COLORS[c] for c in common.LENGTH_ORDER])
    ymax = ax.get_ylim()[1] * 1.2
    ax.set_ylim(0, ymax)
    for name, lo, hi in common.LENGTH_CLASSES:
        if lo > 0:
            ax.axvline(lo * 100, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
        x = (lo * 100 + min(hi * 100, bins[-1])) / 2
        rng_txt = f"< {hi * 100:g} cm" if lo == 0 else (f"≥ {lo * 100:g} cm" if np.isinf(hi) else
                                                       f"{lo * 100:g}–{hi * 100:g} cm")
        ax.text(x, ymax * 0.985, f"{name}\n{rng_txt}", ha="center", va="top", color=INK_2, fontsize=9,
                linespacing=1.3)
    ax.set_xlim(0, bins[-1])
    ax.set_xlabel("median strand length (cm)")
    ax.set_ylabel("hairstyles")
    ax.set_title("Median strand length", loc="left")
    ax.legend(handles=[Patch(color=AVAILABLE_COLOR, label=f"available, not in pool"),
                       *[Patch(color=CLASS_COLORS[c], label=f"pool · {c}") for c in common.LENGTH_ORDER]],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9, labelcolor=INK_2)

    # (2) pool size per class, next to what was available
    ax = axes[1]
    x = np.arange(len(common.LENGTH_ORDER))
    avail_n = [sum(common.length_class(lengths[n]) == c for n in available) for c in common.LENGTH_ORDER]
    pool_n = [sum(common.length_class(lengths[n]) == c for n in pool) for c in common.LENGTH_ORDER]
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, avail_n, w, color=AVAILABLE_COLOR)
    b2 = ax.bar(x + w / 2 + 0.01, pool_n, w, color=[CLASS_COLORS[c] for c in common.LENGTH_ORDER])
    for bars in (b1, b2):
        ax.bar_label(bars, padding=2, color=INK_2, fontsize=9)
    ax.set_xticks(x, list(common.LENGTH_ORDER))
    ax.set_ylim(0, max(avail_n) * 1.15 + 1)
    ax.set_ylabel("hairstyles")
    ax.set_title(f"Pool per class ({len(pool)} of {len(available)})", loc="left")
    ax.legend(handles=[Patch(color=AVAILABLE_COLOR, label="available"),
                       Patch(facecolor="none", edgecolor=INK_MUTED, label="pool (class color)")],
              frameon=False, fontsize=9, labelcolor=INK_2)

    # (3) cases per length pair
    ax = axes[2]
    pairs = common.all_length_pairs(len(cases[0]["heads"])) if cases else []
    counts = [sum(c["length_pair"] == p for c in cases) for p in pairs]
    bars = ax.bar(np.arange(len(pairs)), counts, 0.7, color=INK_2)
    ax.bar_label(bars, padding=2, color=INK_2, fontsize=9)
    ax.set_xticks(np.arange(len(pairs)), [p.replace("+", "\n+ ") for p in pairs], fontsize=9)
    ax.set_ylim(0, max(counts + [1]) * 1.15)
    ax.set_ylabel("cases")
    ax.set_title(f"Cases per length pair ({len(cases)})", loc="left")

    # (4) pair-ratio matrix (upper triangle, unordered pairs): observed share of cases, expected share
    ax = axes[3]
    ax.grid(False)
    classes = list(common.LENGTH_ORDER)
    k = len(classes)
    expected = expected_pair_ratios(pool, lengths)
    if not cases or len(cases[0]["heads"]) != 2:
        ax.axis("off")
        ax.text(0.5, 0.5, "pair matrix: m = 2 only", ha="center", va="center", color=INK_MUTED,
                transform=ax.transAxes)
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("ink", ["#f1f0ec", "#3d3c39"])
        obs = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i, k):
                obs[i, j] = sum(c["length_pair"] == common.length_pair([classes[i], classes[j]])
                                for c in cases) / len(cases)
        ax.imshow(obs, cmap=cmap, vmin=0, vmax=max(np.nanmax(obs), 1e-9))
        for i in range(k):
            for j in range(i, k):
                pair = common.length_pair([classes[i], classes[j]])
                n = int(round(obs[i, j] * len(cases)))
                dark = obs[i, j] / max(np.nanmax(obs), 1e-9) > 0.55
                ax.text(j, i - 0.14, f"{obs[i, j]:.0%}", ha="center", va="center", fontsize=12, fontweight="bold",
                        color=SURFACE if dark else INK)
                ax.text(j, i + 0.22, f"{n} cases\nexpected {expected[pair]:.0%}", ha="center", va="center",
                        fontsize=8, linespacing=1.25, color=SURFACE if dark else INK_2)
        for t in np.arange(k + 1) - 0.5:   # 2px surface gaps between cells
            ax.axhline(t, color=SURFACE, linewidth=3)
            ax.axvline(t, color=SURFACE, linewidth=3)
        ax.set_xticks(range(k), classes)
        ax.set_yticks(range(k), classes)
        ax.tick_params(length=0)
        ax.xaxis.tick_top()
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("Case ratio per length pair (vs expected from pool)", loc="left", pad=22)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def build_case(idx, args, names, lengths, rng):
    case_id = f"case_{idx:04d}"
    case_dir = os.path.join(args.out, "cases", case_id)
    case_seed = int(rng.integers(2**31 - 1))
    crng = np.random.default_rng(case_seed)

    picked = [names[i] for i in crng.choice(len(names), args.m, replace=False)]
    npz_paths = [os.path.join(common.hairstyle_dir(args.dataset_path, n), args.strands_file) for n in picked]
    # the mask only matters where the heads have roots: generate it relative to that region
    region = common.hair_region([common.load_hairstyle(p, root_uv_only=True)["root_uv"] for p in npz_paths],
                                args.res)
    gen = args.mask_generators[int(crng.integers(len(args.mask_generators)))]
    labels, params = generate_labels(gen, crng, region, args.m, min_frac=args.min_frac)

    masks = common.labels_to_masks(labels, args.m)
    mask_files = []
    for k, mk in enumerate(masks, start=1):
        fn = f"mask_{k}.png"
        common.save_mask_png(mk, os.path.join(case_dir, fn))
        mask_files.append(fn)
    common.save_label_preview(labels, region, os.path.join(case_dir, "labels.png"))

    case = dict(
        id=case_id, m=args.m, seed=case_seed,
        heads=[dict(role="base" if i == 0 else f"donor_{i}", name=n, npz_path=os.path.abspath(p),
                    median_length=round(lengths[n], 4), length_class=common.length_class(lengths[n]))
               for i, (n, p) in enumerate(zip(picked, npz_paths))],
        length_pair=common.length_pair([common.length_class(lengths[n]) for n in picked]),
        mask_files=mask_files,          # mask_files[k-1] = head k's region (white)
        mask_layout="scalp_texture", mask_res=args.res,
        mask_generator=gen, mask_params=params,
        hair_region_texels=int(region.sum()),
    )
    common.write_json(case, os.path.join(case_dir, "case.json"))
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("n", type=int, help="number of test cases")
    p.add_argument("--m", type=int, default=2, help="hairstyles per case (head 0 = base); fusion methods "
                                                    "currently support m = 2")
    p.add_argument("--out", default=None, help="output folder (default benchmark/data/fusion_m<m>_n<n>_s<seed>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_path", default=common.DEFAULT_DATASET_PATH)
    p.add_argument("--strands_file", default=common.DEFAULT_STRANDS_FILE,
                   help="strands npz inside each hairstyle folder (full_strands.npz | interpolated_strands.npz)")
    p.add_argument("--mask_generators", default="left_right,front_back,half_plane,polygon",
                   help=f"comma list, one picked uniformly per case; known: {','.join(sorted(MASK_GENERATORS))}")
    p.add_argument("--min_frac", type=float, default=0.15,
                   help="min share of the hair region every head's mask region must cover")
    p.add_argument("--min_hair_length", type=float, default=0.0,
                   help="only sample hairstyles whose median strand length (m, from guide_strands.npz) is at "
                        "least this; about half of the dataset is shorter than 0.05 (buzz cuts barely cross)")
    p.add_argument("--max_hair_length", type=float, default=float("inf"))
    for cls, lo, hi in common.LENGTH_CLASSES:
        rng_txt = f"< {hi * 100:g} cm" if lo == 0 else (f">= {lo * 100:g} cm" if np.isinf(hi) else
                                                       f"{lo * 100:g}-{hi * 100:g} cm")
        p.add_argument(f"--n_{cls}", type=int, default=None,
                       help=f"number of {cls} hairstyles ({rng_txt}) in the pool the cases sample from, drawn at "
                            "random (default: all available; 0 = none)")
    p.add_argument("--res", type=int, default=common.TEXTURE_RES, help="mask resolution (scalp texture)")
    p.add_argument("--force", action="store_true", help="overwrite an existing benchmark folder")
    args = p.parse_args()

    if args.n <= 0:
        p.error("n must be a positive integer")
    if args.m < 2:
        p.error("--m must be >= 2")
    args.mask_generators = [g.strip() for g in args.mask_generators.split(",") if g.strip()]
    unknown = [g for g in args.mask_generators if g not in MASK_GENERATORS]
    if unknown:
        p.error(f"unknown mask generators {unknown}; known: {sorted(MASK_GENERATORS)}")
    args.dataset_path = os.path.abspath(args.dataset_path)
    if args.out is None:
        args.out = os.path.join(common.BENCH_DIR, "data", f"fusion_m{args.m}_n{args.n}_s{args.seed}")
    args.out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(args.out, "benchmark.json")) and not args.force:
        p.error(f"{args.out} already holds a benchmark (use --force to overwrite)")

    all_names = common.list_hairstyles(args.dataset_path, args.strands_file)
    lengths = {n: common.hairstyle_length(args.dataset_path, n) for n in all_names}
    available = [n for n in all_names if args.min_hair_length <= lengths[n] <= args.max_hair_length]
    avail_counts = {c: sum(common.length_class(lengths[n]) == c for n in available) for c in common.LENGTH_ORDER}
    request = {c: getattr(args, f"n_{c}") for c in common.LENGTH_ORDER}
    for c, k in request.items():
        if k is not None and k < 0:
            p.error(f"--n_{c} must be >= 0")
        if k is not None and k > avail_counts[c]:
            p.error(f"--n_{c} {k}: only {avail_counts[c]} {c} hairstyles available "
                    f"(median length in [{args.min_hair_length}, {args.max_hair_length}] m)")
    names = select_pool(available, lengths, request, args.seed)
    pool_counts = {c: sum(common.length_class(lengths[n]) == c for n in names) for c in common.LENGTH_ORDER}
    if len(names) < args.m:
        p.error(f"the pool has only {len(names)} hairstyles, a case needs {args.m}")
    print(f"{len(available)}/{len(all_names)} hairstyles available in {args.dataset_path} "
          f"(median length in [{args.min_hair_length}, {args.max_hair_length}] m): "
          + ", ".join(f"{c} {avail_counts[c]}" for c in common.LENGTH_ORDER))
    print(f"pool: {len(names)} hairstyles, " + ", ".join(f"{c} {pool_counts[c]}" for c in common.LENGTH_ORDER))

    rng = np.random.default_rng(args.seed)
    cases, case_dicts = [], []
    for i in range(args.n):
        c = build_case(i, args, names, lengths, rng)
        case_dicts.append(c)
        fr = ", ".join(f"{f:.2f}" for f in c["mask_params"]["area_fractions"])
        print(f"[{i + 1}/{args.n}] {c['id']}: {' + '.join(h['name'] for h in c['heads'])} ({c['length_pair']}) | "
              f"{c['mask_generator']} (area {fr})")
        cases.append(c["id"])

    common.write_json(dict(
        version=BENCHMARK_VERSION, created=common.now_str(), n=args.n, m=args.m, seed=args.seed,
        dataset_path=args.dataset_path, strands_file=args.strands_file,
        min_hair_length=args.min_hair_length,
        max_hair_length=None if np.isinf(args.max_hair_length) else args.max_hair_length,
        length_standard=common.length_standard(),
        n_hairstyles_available=len(available), available_class_counts=avail_counts,
        pool_request={c: k for c, k in request.items()},          # None = all available of that class
        pool_class_counts=pool_counts,
        length_class_counts=pool_counts,                          # (v1 name, kept for older readers)
        hairstyles=names,                                         # the pool
        pool=[dict(name=n, median_length=round(lengths[n], 4), length_class=common.length_class(lengths[n]))
              for n in names],
        case_length_pairs={pp: sum(c["length_pair"] == pp for c in case_dicts)
                           for pp in common.all_length_pairs(args.m)},
        case_length_pair_ratios={pp: round(sum(c["length_pair"] == pp for c in case_dicts) / args.n, 4)
                                 for pp in common.all_length_pairs(args.m)},
        **(dict(expected_length_pair_ratios={pp: round(r, 4) for pp, r in expected_pair_ratios(names, lengths).items()})
           if args.m == 2 else {}),
        mask_res=args.res, mask_layout="scalp_texture", mask_generators=args.mask_generators,
        min_frac=args.min_frac, cases=cases,
    ), os.path.join(args.out, "benchmark.json"))
    plot_length_distribution(os.path.join(args.out, "length_distribution.png"), lengths, available, names,
                             case_dicts, f"{os.path.basename(args.out)}: {args.n} cases, m = {args.m}, seed {args.seed}")
    if args.m == 2:
        exp = expected_pair_ratios(names, lengths)
        print("length pairs (cases / ratio / expected from pool): " + ", ".join(
            f"{pp} {sum(c['length_pair'] == pp for c in case_dicts)} / "
            f"{sum(c['length_pair'] == pp for c in case_dicts) / args.n:.0%} / {exp[pp]:.0%}" for pp in exp))
    print(f"benchmark written to {args.out} (length distribution: length_distribution.png)")


if __name__ == "__main__":
    main()
