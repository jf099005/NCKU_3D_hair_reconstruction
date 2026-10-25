# Hair fusion benchmark

This benchmark evaluates the per-strand fusion of `inference/img2hair_tsai_1.py` and its grid score on
ground-truth DiffLocks dataset hairstyles. It skips the image → hair reconstruction, so the fusion is the
only thing being measured. It has two parts: **build** (sample the test cases) and **run** (fuse and score
them).

Use an env with numba, mediapipe and NATTEN (`hair_exp`; `difflocks` has no numba). `bench.sh` uses
`$PYTHON`, which defaults to hair_exp.

```bash
# build n cases (m = 2 hairstyles + 1 mask each); --min_hair_length 0.12 skips buzz cuts
bash benchmark/bench.sh build 50 --seed 0 --min_hair_length 0.12 --out benchmark/data/m2_n50
# or fix the pool size per length class (random hairstyles of each class; default all, 0 = none)
bash benchmark/bench.sh build 50 --seed 0 --n_short 15 --n_mid 25 --n_long 25 --out benchmark/data/m2_n50_bal
# run all methods on every case, then summarize + render the extremes (resumable; --max_strands 20000 for a quick pass)
bash benchmark/bench.sh run benchmark/data/m2_n50 --run_name full
# re-aggregate a run: summary.{csv,json,md} + report.html (open in a browser); renders missing extremes
bash benchmark/bench.sh summarize benchmark/data/m2_n50/runs/full [--no_render]
# render any other case afterwards (reuses the saved keep masks, no FM rerun; ~35 s per render)
bash benchmark/bench.sh render benchmark/data/m2_n50/runs/full --cases case_0000,... [--methods tsai_joint_fm]
```

## Hair length classes and the report

Every hairstyle is classified by its median strand length (from its small `guide_strands.npz`,
`common.LENGTH_CLASSES`). On a head about 16 cm wide:

| class | median strand length | typical style |
|---|---|---|
| short | < 5 cm | buzz / crew cut / pixie, lies on the scalp |
| mid | 5 – 15 cm | covers the ears, down to about the chin |
| long | ≥ 15 cm | past the chin, shoulder length and longer |

A case's **length pair** is the unordered combination of its heads' classes (`short+short`, `short+mid`,
`short+long`, `mid+mid`, `mid+long`, `long+long`). `build` writes each head's `median_length` and
`length_class` and the case's `length_pair` into `case.json`. Older benchmarks are classified when they are
summarized.

`summarize` (which also runs at the end of `run`) reports **averages only**, for the whole benchmark and
for each length pair:
- mean ± std of the key metrics per method
- the paired comparison against `mask_only` (mean delta, relative delta, win rate)
- each group's lowest and highest `grid_score.mean_all` case of `tsai_joint_fm` (`--select_method`), with
  the drawn mask, every method's multi-view render, and the flux / vector-field figures of
  `img2hair_tsai_1.debug_field_analysis` (`analysis.py`)

The analysis uses fused "before FM" = `mask_only`'s result and "after FM" = `tsai_joint_fm`'s. It shows the
|flux| map and the flow vector field of base, donor, before and after, plus the grid-score map. The
extremes are rendered and analysed automatically, about 1 min each, and existing results are reused.
`--no_render` and `--no_analysis` skip these steps.

The report has several pages, all in the run folder:
- `report.html` is the index: length classes, an overview of grid score by pair, and the all-cases group.
- `report_<pair>.html` (e.g. `report_short-long.html`) has one page per length pair.

The pages link images by relative path, so keep them inside the run folder.

## Benchmark data (`build_benchmark.py`)

```
<out>/benchmark.json                 settings, pool (hairstyles + median length + class), class counts
                                     (available / requested / pool), cases per length pair, case ids
<out>/length_distribution.png        median strand length: available vs pool (by class), pool per class,
                                     cases per length pair, and a short/mid/long x short/mid/long matrix of
                                     the case ratio per pair vs the ratio expected from the pool (m = 2)
<out>/cases/case_XXXX/case.json      heads (dataset name, absolute npz path, median_length, length_class; head 0 = base),
                                     length_pair, mask generator + params, seed
<out>/cases/case_XXXX/mask_<k>.png   region of head k (white), k = 1..m-1  (same convention as the fusion configs)
<out>/cases/case_XXXX/labels.png     preview (head colors, outside the hair region dark)
```

The builder does not copy hairstyles: each case only points to `generated_hairstyles/<name>/full_strands.npz`.

Cases sample their hairstyles from a **pool**. The pool starts as the dataset hairstyles within
`--min/--max_hair_length`; `--n_short`, `--n_mid` and `--n_long` then draw that many random hairstyles of each
class (default: all of the class; `0` leaves the class out; asking for more than are available is an error).
The pool is drawn from `--seed` too, so the same command rebuilds the same benchmark. The cases still pick their m
hairstyles uniformly from the pool, so the pool's class mix sets the expected mix of length pairs: C(n_a, 2) / C(N, 2) for a+a and
n_a·n_b / C(N, 2) for a+b (`expected_length_pair_ratios` in benchmark.json, next to the actual
`case_length_pair_ratios`).

Masks use the **scalp-texture layout** (256×256; columns run left↔right on the head, rows front↔back), the
same layout as the fusion masks of `img2hair_tsai_1.py`. On load, a dataset hairstyle's `root_uv` is
converted to the DiffLocks output convention (row, col) / 256. This uses the same mapping as
`data_processing/create_scalp_textures.py`: row = 1 − v, col = u, with u flipped when `mirror_hair` is set
(`common.gt_root_uv_to_texture_uv`).

**Mask generators** (`masks.py`): `left_right`, `front_back`, `half_plane` (parallel cuts at a random
angle) and `polygon` (a random convex polygon per donor). A generator returns an int label map (0 = base,
k = head k) together with its parameters. Each one is a single function registered with
`@register_mask("name")`. `generate_labels` redraws until every head covers `[min_frac, max_frac]` of the
hair region, which is the union of the case's root texels.

## Running (`run_benchmark.py`)

For each case and method:

```
<bench>/runs/<run_name>/cases/case_XXXX/<method>/
    result.json         status, timings, method info (FM objectives before/after), all metrics
    maps.npz            (n, n) maps of each metric (grid score, |flux|, crossing pairs, bald cells) + per-head keep masks
    labels_refined.png  roots.png  [render/render_multiview.png: only the extremes of each length pair]
<bench>/runs/<run_name>/summary.{csv,json,md}, report.html
```

**Methods** (`fusion_methods.py`, `@register_method(name, supported_m=...)`):
- `mask_only`: no refinement; each strand follows the drawn mask at its root. This is the reference.
- `tsai_joint_fm`: `update_joint_mask` with `img2hair_tsai_1.JOINT_FM_KWARGS`, the script's own settings
  imported directly. It supports m = 2 only.
- `tsai2_joint_fm_trim` (only with `run --tsai2`): `inference/img2hair_tsai_2.py`, i.e. the same joint FM and
  then a trimming step (`joint_contour.trim.joint_trim`). It cuts `--trim_percent` % (default 0.5) of all
  strands of both heads: the visible ones crossing the other head's visible strands the most (projection
  crossings, the FM's own rule). The FM then runs once per case and both tsai methods share it, with the
  same `tsai_joint_fm` result as without the flag. Its `method_seconds` is the trimming alone
  (`method_info.fm_seconds` has the shared FM). Without `--tsai2` nothing changes.
  ```bash
  bash benchmark/bench.sh run <bench> --tsai2 [--trim_percent 0.5]
  ```
  A run keeps one `trim_percent`; compare other values in separate `--run_name`s.
- `--boundary_fm`: the joint FM of the tsai methods runs as Boundary FM (`JOINT_FM_KWARGS` + `boundary=True`,
  `joint_contour/boundary_fm.py`); `method_info.joint_fm_kwargs` records it. Method names stay the same and
  `run_config.json` pins `boundary_fm`, so compare against plain FM in a separate `--run_name`:
  ```bash
  bash benchmark/bench.sh run <bench> --run_name bfm --boundary_fm [--tsai2]
  ```
  Without the flag nothing changes.

**Metrics** (`metrics.py`, `@register_metric`) are computed the same way for every method on a 32×32
analysis grid (`--grid_n`):

| metric | meaning |
|---|---|
| `grid_score.*` | convex-hull overlap of base vs donor strands per cell (`mask_analysis.cal_grid_score`). `mean_all` equals `mean_hull_score` in tsai's `stats.json`. `mixed_mean` is over cells crossed by both heads. `frac_above` is the share of cells with score > 0.1. `total` is the summed score; `per_black_area` / `per_white_area` divide it by the black (donor) / white (base) area, the cells of the fused grid mask (`fused_grid_mask.png`) where the donor's kept strands are a strict majority / the other cells crossed by kept strands (`black_cells`, `white_cells`) |
| `crossings.*` | column-rule crossing pairs (kept strands of head i × head j per cell) and conflict cells |
| `coverage.bald_frac` | cells that had roots in some input but have no kept root (holes) |
| `flux.abs_mean` | mean \|3D flux\| of the fused hair's flow field |
| `strands.*` | kept strands per head, donor share, fraction of strands whose label differs from the drawn mask |

The tsai methods also record the size of the joint FM's overlap matrix (`overlap="projection"`) in
`result.json` → `method_info.overlap_matrix` (summary columns `overlap_matrix.*`): `n_a` × `n_b` base × donor
strands, `nnz` = pairs whose projections cross in at least one column (the matrix's 1 elements), `density` =
nnz / (n_a · n_b) (sparsity = 1 − density), `weight_sum` = crossing columns summed over the pairs, and
`node_pairs` / `n_nodes` for the FM's cell nodes. It is read from the graphs `update_joint_mask` builds anyway
(`fusion_methods._record_overlap_matrix` wraps `joint_contour.core`'s builders during the call), so joint_contour
is unchanged and the FM costs no extra time or memory.

Cost per case with all strands (~95k per head): about 5–6 min, mostly the joint FM. The GPU is used only for
the strand → grid-cell membership queries. RAM depends on the case: `tsai_joint_fm` (overlap="projection")
stores every base × donor strand pair whose projections cross, so two dense, heavily overlapping heads can
need more than 28 GB. With `--max_strands 20000` a case takes about 1 min and far less memory.

**Memory guard.** Every case runs in its own child process, which the runner watches. The child is killed
when its RAM passes `--max_mem_gb` (default 70% of the machine) or the system's free memory drops under
`--min_free_gb` (4), which keeps the desktop responsive. The method it was running is recorded as `status: skipped_oom`, and the run continues
with the next case. The child also makes itself the kernel OOM killer's first target, so the desktop and
editor survive. Reruns skip `skipped_oom` results like finished ones; to try them again (e.g. with fewer
strands, in a separate run since `--max_strands` must match):
```bash
bash benchmark/bench.sh run <bench> --retry_skipped                       # same settings, e.g. a higher --max_mem_gb
bash benchmark/bench.sh run <bench> --run_name small --cases case_0022 --max_strands 40000
```
The report counts them under "failed" (hover for the cases and their status). `--no_isolate` runs
everything in one process without the guard (the old behavior). `bench.sh` sets `PYTHONUNBUFFERED=1`, so a
log written with `nohup ... > run.log` is up to date.

## Extending to m > 2

Everything except `tsai_joint_fm` already handles m heads:
- `build --m 3` writes 2 masks per case.
- the mask generators make m − 1 cuts or polygons.
- the pairwise metrics are summed or averaged over head pairs.
- renders use a gray ramp.

For m > 2 the runner currently skips `tsai_joint_fm`, because it declares `supported_m=(2,)`. To add an
m > 2 fusion algorithm, register a new method that maps `(heads, labels, ctx)` to
`FusionResult(keep=[...per head...])`.
