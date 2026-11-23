# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of the official DiffLocks codebase (single image → strand-based 3D hair via diffusion), extended for NCKU research on **multi-image hairstyle fusion** and **hair color reconstruction**. The upstream code (`models/`, `k_diffusion/`, `modules/`, `data_loader/`, `train_*.py`, `inference/img2hair.py`, `inference/npz2blender.py`) is the base; most active work is in `inference/img2hair_*` variants, `experiments/`, and `hair_color_reconstruction/`. Many comments and shell-script messages are in Traditional Chinese.

There is no test suite, linter or build step. Changes are checked by running the pipelines and looking at the renders.

## Environment

- Conda env `difflocks` (see `hair_color_reconstruction/run_demo.sh`); deps in `requirements.txt` (torch 2.5.0 pinned). You also need to install the custom CUDA kernels **NATTEN** (a vendored copy is in `NATTEN/`) and **FlashAttention-2**. The diffusion model runs under bf16 autocast because flash attention needs it.
- Blender **4.1.1** is the intended version, at `/home/kyh/blender/blender` (this path is hardcoded in several scripts, e.g. `BLENDER_PATH` in `inference/img2hair_tsai_1.py` and `benchmark/common.py`). On the RTX 4060 Ti machine that path is Blender **5.0.0**, where OptiX can crash on some hair scenes (`Misaligned address in CUDA queue ... integrator_intersect_shadow`). Rerun those with `CYCLES_DEVICE=CUDA`, which `experiments/gpu_render.py` reads (default `OPTIX`). Blender scripts (`inference/npz2blender*.py`, `inference/render_multiview_blender.py`, `experiments/gpu_render.py`, `hair_color_reconstruction/render_frontview_blender.py`) run inside Blender's own Python: invoke them as `blender --background --python <script> -- <args>`, never with plain `python`.
- Checkpoints go in `checkpoints/{strand_vae,difflocks_diffusion,rgb2material}` (`./download_checkpoints.sh`). The dataset lives outside the repo at `../DiffLocks_Dataset/difflocks_dataset` (raw) and `../DiffLocks_Dataset/difflocks_dataset_processed`.
- Scripts use paths relative to the repo root (`./checkpoints/...`, `./configs/...`), so run them from the repo root. The exception is `experiments/`, which has symlinks (`checkpoints`, `configs`, `inference`, and `samples` → `../../../P76134723/difflocks/samples`) so its scripts run from inside `experiments/`.

## Common commands

```bash
# Single-image inference -> outputs_inference/difflocks_output_strands.npz (+ .blend if --blender_path)
./inference_difflocks.py --img_path=<img> --out_path=./outputs_inference/ [--blender_path=/home/kyh/blender/blender]

# Hair-color comparison: DiffLocks rgb2material color vs. median-pixel color, multi-view renders side by side
./run_difflocks.sh <img_path> [out_path] [blender_path]

# Multi-image fusion (edit SCRIPT / input_config at top of script; config = {"input_file_paths": [...], "mask_file_paths": [...]})
cd experiments && bash run_fusion.sh [--individual] [--no-keep-model]
cd experiments && bash rendering.sh <dir_with_blender_scene.blend> <render_out>
# Comparison report (one self-contained HTML, one row per variant; run_fusion.sh writes <output_path>/report.html)
cd experiments && python make_report.py <output_path_or_method_dir> [...] [-o compare.html]

# Hair-color fine-tuning end to end (latents -> fine-tune -> eval -> visual demo); edit variables at top first
cd hair_color_reconstruction && ./run_demo.sh

# Fusion benchmark on GT dataset hairstyles (env hair_exp): build n cases, then run tsai_joint_fm vs mask_only
bash benchmark/bench.sh build <n> --min_hair_length 0.12 --out benchmark/data/<name>
bash benchmark/bench.sh run benchmark/data/<name> [--max_strands 20000] [--tsai2 --trim_percent 0.5]   # report.html: averages per hair-length pair; see benchmark/README.md
bash benchmark/bench.sh summarize <run_dir>                          # rebuild summary.* / report.html (reuses renders + analyses)
bash benchmark/bench.sh render <run_dir> --cases <ids> [--methods m]  # render more cases without rerunning the fusion

# Training (from upstream README)
./train_strandsVAE.py --dataset_path=<DATASET_PATH> --exp_info=<NAME> [--model_size small|tiny|base --normalization custom|difflocks]
./train_rgb2material.py --dataset_path=<DATASET_PATH> ...
accelerate launch ./train_scalp_diffusion.py --config ./configs/config_scalp_texture_conditional.json \
    --dataset_path=<DATASET_PATH> --dataset_processed_path=<DATASET_PATH_PROCESSED> --name <NAME> ...
```

Before training, prepare the dataset with `data_processing/uncompress_data.py`, then `create_chunked_strands.py`, `create_latents.py` (DINOv2 latents) and `create_scalp_textures.py` (needs the strand VAE checkpoint). Logs go to `./tensorboard_logs`; training outputs go to `./out_training/`.

## Architecture: the inference pipeline

`DiffLocksInference` in `inference/img2hair.py` is the core. Every `img2hair_*` variant copies and extends it:

1. **Face crop**: MediaPipe FaceLandmarker finds landmarks, then `crop_face` crops to 770×770 (the multiple of 14 closest to that size, to match the DINOv2 patch size).
2. **Conditioning**: DINOv2 ViT-L/14-reg (`torch.hub`) produces patch tokens (reshaped to BCHW) and the CLS token, passed as `extra_args['latents_dict']['dinov2']`.
3. **Diffusion**: a k-diffusion hierarchical transformer (`k_diffusion/`, config `configs/config_scalp_texture_conditional.json`) is sampled with CFG (`utils/diffusion_utils.sample_images_cfg`). It outputs a **scalp texture**: per-texel strand latent channels plus a final **density** channel.
4. **Strand sampling and decoding**: `utils/strand_util.sample_strands_from_scalp_with_density` places strand roots on the scalp mesh (`data_loader/difflocks_bodydata/scalp.ply`) in proportion to density. `models/strand_codec.StrandCodec` decodes each root's latent into 256 vertices in TBN (tangent) space, which `data_loader/mesh_utils.tbn_space_to_world` converts to world space.
5. **Material**: `models/rgb_to_material.RGB2MaterialModel` maps DINOv2 features to 11 Principled Hair BSDF params (`hair.json`). Only melanin amount/redness (and the root_darkness_* params) set color. The research alternative is a median RGB over MediaPipe hair-segmentation pixels (`inference/extract_median_hair_color.py`, `hair_color_reconstruction/extract_hair_color_pixels.py`).
6. **Export**: the output is `difflocks_output_strands.npz`. A Blender script turns it into `.blend` / alembic / renders.

### Multi-image fusion (`inference/img2hair_tsai_1.py`, `img2hair_kung_*`)

The `img2hair_kung_*` scripts generate one scalp texture per input image, then blend the textures **in scalp-texture (latent + density) space** using masks from `mask_file_paths` before decoding a single fused hairstyle. `img2hair_tsai_1.py` works per strand instead: it takes exactly 2 images (base, donor) and 1 mask, reconstructs each hairstyle on its own, turns the mask into per-strand masks refined jointly against both heads (minimizing crossings between the two overlaid heads), and composes the fused hair directly from the kept strands (no texture blending or re-decoding). With `--align_hairline` (`run_fusion.sh --align-hairline`), it first runs diffusion only for both images, aligns the two scalp textures with kung's `align_hairline_by_shifting`, and then decodes each one (cached in `individual_strands_aligned/`) before the per-strand fusion. Fusion masks are always in the **scalp-texture layout** (row, col of the 256×256 texture, the layout DiffLocks' `root_uv` = (row, col) / 256 uses), for both script families: `img2hair_tsai_1.py` looks each strand up at its texel (`root_texels`) and passes it to `update_joint_mask` as `rowcol_a/rowcol_b`, so a mask splits the head the same way in both. In this layout, image columns run left↔right on the head. Base strands are forced white and donor strands black, written as `strand_colors` into the npz, which `npz2blender_kung.py` applies per strand ahead of `color_map`. It does this by importing `highlighting.generate_from_3D_models.joint_contour` from the sibling repo `../highlighting`, adding the parent folder `P76154862/` to `sys.path`. Both repos must therefore stay side by side. `inference/fusion_refine/` holds the refinements developed in `benchmark/` (adaptive lam + retention guard, depth-aware crossings, lateral combing, faster membership; see its README); the scripts do not use it yet. `img2hair_tsai_2.py` imports tsai_1 (no copy) and adds a trimming step after the joint FM: `joint_contour.trim.joint_trim` cuts `--trim_percent` % (default 0.5) of all strands, the visible ones crossing the other head most. The `*_kung*` variants write per-strand color maps into the npz, which `inference/npz2blender_kung*.py` reads.

### Fusion benchmark (`benchmark/`)

See `benchmark/CLAUDE.md` for the conventions and internals that matter when changing the benchmark code. The benchmark skips image → hair entirely and runs on GT dataset hairstyles, so it measures only the fusion step. Its `tsai_joint_fm` method imports `img2hair_tsai_1.update_joint_mask` and `JOINT_FM_KWARGS` directly, so editing those changes what the benchmark measures (`--tsai2` does the same with tsai_2's trimming). Methods and metrics are registered with `@register_method` / `@register_metric` in `fusion_methods.py` / `metrics.py`. Notes:
- Runs can be resumed: (case, method) pairs that already have a `result.json` are skipped unless you pass `--overwrite`. Use a new `--run_name` after changing the algorithm.
- Each case runs in a child process under a memory guard. A full-strand case (~95k strands per head) peaks around 12 GB of RAM, almost all of it the projection-crossing graph (`update_contour/overlap.weighted_pairs` in `../highlighting`, which calls `malloc_trim` so glibc's per-thread arenas do not keep the per-column arrays; without it the peak was ~19 GB). When the guard kills the child, the result gets status `skipped_oom`. Those cases are not retried on a rerun unless you pass `--retry_skipped` (add `--max_strands` to make them fit).
- `run` calls `summarize.py` at the end. Summarize renders the lowest- and highest-scoring case of every hair-length group and runs `debug_field_analysis` on them via `analysis.py`, which passes `save_field_data=False` because each `field_data.npz` would be ~1.5 GB and the report never reads it. The per-method averages in `summary.md` include degenerate results (one head almost vanishes, `min_head_retention` < 0.1) and cover different sets of cases when some are `skipped_oom`. Compare methods with the paired table.
- `bench.sh` sets `PYTHONUNBUFFERED=1`, so a `nohup` log shows progress as it happens.

`hair_color_reconstruction/` reuses `create_dataloaders` / `compute_loss` / `prepare_gt_batch` from `train_rgb2material.py`. Keep those function signatures stable.
