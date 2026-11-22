# CLAUDE.md — benchmark/

The fusion benchmark. It runs the per-strand fusion of `inference/img2hair_tsai_1.py` (and `img2hair_tsai_2.py`)
on ground-truth DiffLocks dataset hairstyles instead of reconstructed ones, so it measures only the fusion step
and its grid score. `README.md` is the user guide (commands, outputs, metrics). This file holds what you need to
know before changing the code.

## Running

Everything goes through `bench.sh`, which uses env `hair_exp` (`$PYTHON` overrides it). That env has numba,
mediapipe and NATTEN; `difflocks` has no numba. `bench.sh` also sets `PYTHONUNBUFFERED=1`.

```bash
bash benchmark/bench.sh build <n> [--seed 0] [--n_short 15 --n_mid 25 --n_long 25] [--min_hair_length 0.12] --out benchmark/data/<name>
bash benchmark/bench.sh run benchmark/data/<name> [--run_name run] [--max_strands 20000] [--tsai2 --trim_percent 0.5] [--boundary_fm] [--no_render]
bash benchmark/bench.sh summarize <run_dir> [--no_render]          # summary.* + report.html; reuses renders / analyses
bash benchmark/bench.sh render <run_dir> --cases <ids|all> [--methods m]
```

For a quick check, use a small benchmark under the scratchpad and `--max_strands 3000 --no_render --no_figures`
(about 1 min per case). A full case (~95k strands per head) takes about 5–6 min. `data/` is gitignored. Do not
write test output into the user's `data/` benchmarks, and do not start full runs while the user's run is going:
they share the GPU and RAM.

## Files

| file | role |
|---|---|
| `common.py` | paths (puts repo, `inference/` and the parent `P76154862/` on `sys.path`), dataset loading, root_uv conversion, length classes, mask I/O |
| `masks.py` | mask generators, `@register_mask(name)`: `fn(rng, region, m, **kw) -> (labels, params)` |
| `build_benchmark.py` | pool selection per length class, case sampling, `length_distribution.png` |
| `fusion_methods.py` | methods, `@register_method(name, supported_m)`: `fn(heads, labels, ctx) -> FusionResult(keep, refined_labels, info)` |
| `metrics.py` | metrics on a 32×32 analysis grid, `@register_metric(name)` |
| `run_benchmark.py` | per-case loop, memory guard (child process per case), `result.json` / `maps.npz`, then calls summarize |
| `summarize.py`, `report.py` | averages per length pair, paired comparison vs `mask_only`, extremes, HTML report |
| `analysis.py` | `tsai.debug_field_analysis` on the extreme cases (`mask_only` = before, `tsai_joint_fm` = after) |
| `render_results.py` | renders from saved keep masks with Blender (`experiments/gpu_render.py`); `--optical` = highlighting's optical preset (full-strand runs) |
| `adaptive_fm.py`, `strand_edit.py` | experimental methods (`afm_*`, `edit_*`), registered at the end of `fusion_methods.py`, opt-in via `--methods` |
| `dev_table.py`, `compare_page.py` | quick per-case tables of a dev run; comparison images (originals, 2D / scalp mask, optical renders) |

Experiment tracking for method development: `LOG.md` (append only), `SUMMARY.md` (current status), `checkpoints.md` (usable versions).
A method may return `FusionResult.heads` (moved strands); metrics and renders then use them (`edited_heads.npz`). `view_mix` is an
opt-in metric (`--metrics ...,view_mix`) that sees layering, unlike the column grid score.

## Conventions that must hold

- **Masks are in the scalp-texture layout** (256×256, row/col; columns run left↔right on the head), the same as
  the fusion configs of `img2hair_tsai_1.py`. Dataset `root_uv` is converted on load by
  `common.gt_root_uv_to_texture_uv`: row = 1 − v, col = u, with u flipped when the metadata sets `mirror_hair`.
  Strands look up their label at `common.root_texels`. Do not use joint_contour's scalp-grid layout for masks.
- **Never copy dataset hairstyles.** `case.json` stores the absolute npz path only.
- **The benchmark imports the algorithm, it does not copy it.** `tsai_joint_fm` calls
  `img2hair_tsai_1.update_joint_mask` with `img2hair_tsai_1.JOINT_FM_KWARGS`, and `tsai2_joint_fm_trim` uses
  `img2hair_tsai_2.trim_joint_result`. Editing those scripts changes what the benchmark measures. After an
  algorithm change, use a new `--run_name`: old results would otherwise be reused as "done".
- **Flags off = old behavior.** Without `--tsai2`, `tsai_joint_fm` must take its original code path (the
  `else` branch in `fusion_methods.tsai_joint_fm`). With `--tsai2`, both tsai methods share one FM per case
  (`_shared_joint_fm`, cached in `ctx` and cleared at the end of each case). Its `tsai_joint_fm` keep masks must
  be identical to the flag-off ones; check this whenever you touch either path. `--boundary_fm` likewise:
  without it `fusion_methods._joint_fm_kwargs` returns the very same `JOINT_FM_KWARGS` object and ctx has no
  `boundary_fm` key.
- **Results must stay reproducible from the files.** Heads are loaded with `max_strands` and seed
  `case_seed + head index`. `maps.npz` stores `keep_<i>` so renders can be rebuilt without rerunning FM.
  `run_config.json` pins `max_strands`, `grid_n`, (with `--tsai2`) `trim_percent` and `boundary_fm`; a resume with
  different values is refused.
- **m > 2.** Everything except the tsai methods handles m heads. Keep new code m-generic (mask generators make
  m − 1 cuts, pairwise metrics sum over head pairs); declare `supported_m` on m = 2-only methods.

## Memory and the guard

The heavy part is RAM, not GPU memory. With `overlap="projection"`, the joint FM stores every base × donor
strand pair whose projections cross (`joint_contour/overlap.py`). This grows with the square of the strand
count and with how much the two heads overlap. Two variants of the same base hairstyle (e.g. `base_43_idx_*`
twice) are the worst case. The GPU only runs the strand → cell membership queries (on the CPU those are ~37×
slower).

`run_benchmark.py` runs each case in a child process (`--_worker`), polls its RSS from the parent and kills it
above `--max_mem_gb` (default 70% of RAM) or when available memory drops under `--min_free_gb` (default 4). The
running method then gets `status: skipped_oom`. Reruns skip those unless `--retry_skipped`. The child sets its
own `oom_score_adj` to 1000, so a kernel OOM kill hits the benchmark, not the desktop. `--no_isolate` runs
in-process without the guard. Any new result written by a child gets `case_peak_rss_gb`.

## Report semantics

- Length classes from the median strand length of `guide_strands.npz`: short < 5 cm, mid 5–15 cm, long ≥ 15 cm
  (`common.LENGTH_CLASSES`). A case's length pair is the unordered class combination.
- The report shows averages only, never per-case scores, plus the lowest / highest `grid_score.mean_all` case of
  `tsai_joint_fm` per group (rendered + analysed).
- `degenerate` = `min_head_retention` < 0.1 (a head practically vanished). Degenerate results count in the
  averages but are picked as extremes only when a group has nothing else (then flagged in the report).
- Methods can cover different case sets when some are `skipped_oom` or failed; the paired table vs `mask_only`
  is the fair comparison.
- `analysis.py` passes `save_field_data=False`: each `field_data.npz` would be ~1.5 GB and nothing reads it.

## Rendering

Renders run Blender at `common.BLENDER_PATH` with `experiments/gpu_render.py` (base white, donor black). If OptiX
crashes on a hair scene (`Misaligned address in CUDA queue`), rerun with `CYCLES_DEVICE=CUDA`.
