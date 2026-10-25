"""img2hair_tsai_2: img2hair_tsai_1 + a trimming step after the joint FM.

Everything up to the joint FM is img2hair_tsai_1 (imported, not copied: same DiffLocksInference, masks,
JOINT_FM_KWARGS, debug analysis, compose / Blender export). After update_joint_mask, joint_contour's
joint_trim drops the visible strands that still cross the most visible strands of the other head: the
number cut is --trim_percent % of all strands of both heads (default 0.5 %), counted the same way as the FM
objective (projection crossings with JOINT_FM_KWARGS["overlap"] == "projection", shared columns otherwise).
Trimmed strands are left out of the fused hair (and drawn gray in the debug renders).

    python inference/img2hair_tsai_2.py --input_config <cfg.json> --output_path <out> [--trim_percent 0.5]
           [--align_hairline] [--blender_path ...] [--seed 5]

Outputs as img2hair_tsai_1, plus <out>/trim.json (percent, strands cut per head).
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import img2hair_tsai_1 as tsai_1  # noqa: E402  (also puts P76154862/ on sys.path for highlighting)
from img2hair_tsai_1 import (  # noqa: E402
    BASE_STRAND_COLOR, BLENDER_PATH, DONOR_STRAND_COLOR, EXCLUDED_STRAND_COLOR, JOINT_FM_KWARGS, ROOT,
    DiffLocksInference, ScalpGrid, compose_hair_per_strand, create_blender_file, debug_field_analysis,
    debug_render_individual_hairstyle, ensure_aligned_strands_npz, ensure_individual_strands_npz,
    load_and_resize_mask, load_strands_npz, root_texels, save_grayscale_mask, update_joint_mask,
)
from highlighting.generate_from_3D_models.joint_contour.grid import (  # noqa: E402
    JointMembership, strand_cell_membership,
)
from highlighting.generate_from_3D_models.joint_contour.overlap import joint_projection_edges  # noqa: E402
from highlighting.generate_from_3D_models.joint_contour.trim import joint_trim  # noqa: E402

# share of all strands (base + donor) cut by the trimming step, in percent
TRIM_PERCENT = 0.5


def trim_joint_result(result, grid, membership_a, membership_b, positions_a, positions_b,
                      trim_percent=TRIM_PERCENT, fm_kwargs=JOINT_FM_KWARGS, verbose=True):
    """Trimming after the joint FM (in place on update_joint_mask's result dict).

    Cuts round(trim_percent / 100 * (S_a + S_b)) visible strands, those with the most crossings with the
    visible strands of the other head (joint_contour.trim.joint_trim; strands without any crossing are
    never cut, so fewer may go). Crossings use the FM's own rule: projection intersections when
    fm_kwargs["overlap"] == "projection" (the edges are recomputed here, update_joint_mask does not return
    them), otherwise shared columns. membership_a / membership_b: the strand_cell_membership the FM used.

    Adds to result: visible_a_fm / visible_b_fm (the FM's result), trim_a / trim_b (cut strands) and
    trim_info; visible_a / visible_b become the kept strands after trimming. Returns result."""
    n_a, n_b = membership_a.shape[1], membership_b.shape[1]
    max_cut = int(round(trim_percent / 100.0 * (n_a + n_b)))
    result["visible_a_fm"], result["visible_b_fm"] = result["visible_a"].copy(), result["visible_b"].copy()
    trim = np.zeros(n_a + n_b, dtype=bool)
    if max_cut > 0:
        membership = JointMembership.from_heads(membership_a, membership_b, grid.n)
        edges = None
        if fm_kwargs.get("overlap", "cell") == "projection":
            edges = joint_projection_edges(grid, membership, positions_a, positions_b, show_progress=verbose,
                                           **fm_kwargs.get("overlap_kwargs", {}))
        state = np.concatenate([result["state_a"], result["state_b"]])
        trim = joint_trim(state, membership, max_cut=max_cut, edges=edges)
        del edges
    result["trim_a"], result["trim_b"] = trim[:n_a], trim[n_a:]
    result["visible_a"] = result["visible_a_fm"] & ~result["trim_a"]
    result["visible_b"] = result["visible_b_fm"] & ~result["trim_b"]
    result["trim_info"] = dict(trim_percent=trim_percent, max_cut=max_cut,
                               cut_base=int(result["trim_a"].sum()), cut_donor=int(result["trim_b"].sum()),
                               overlap=fm_kwargs.get("overlap", "cell"))
    return result


def joint_fm_with_trim(positions_a, root_uv_a, positions_b, root_uv_b, mask, grid, rowcol_a, rowcol_b,
                       trim_percent=TRIM_PERCENT, fm_kwargs=JOINT_FM_KWARGS, device="cuda", verbose=True):
    """update_joint_mask exactly as img2hair_tsai_1 calls it, with the strand/cell memberships computed
    here (the same call update_joint_mask makes internally) so the trimming can reuse them, then
    trim_joint_result."""
    membership_a = strand_cell_membership(grid, positions_a, device=device, show_progress=verbose)
    membership_b = strand_cell_membership(grid, positions_b, device=device, show_progress=verbose)
    result = update_joint_mask(
        positions_a, root_uv_a, positions_b, root_uv_b, mask,
        device=device, grid=grid, method_kwargs=fm_kwargs, verbose=verbose,
        membership_a=membership_a, membership_b=membership_b, rowcol_a=rowcol_a, rowcol_b=rowcol_b,
    )
    return trim_joint_result(result, grid, membership_a, membership_b, positions_a, positions_b,
                             trim_percent=trim_percent, fm_kwargs=fm_kwargs, verbose=verbose)


def refine_mask_per_strand(mask_file_path, base_img_path, donor_img_path, difflocks, out_path, mask_res,
                           align_hairline=False, trim_percent=TRIM_PERCENT):
    """img2hair_tsai_1.refine_mask_per_strand + trimming (joint_fm_with_trim). Returns (base, donor, result);
    result["visible_a"] / ["visible_b"] are the strands kept after trimming."""
    raw_mask = load_and_resize_mask(mask_file_path, mask_res, mask_res, "cuda", blur_sigma=0.0)
    raw_mask_np = raw_mask.squeeze(0).squeeze(0).cpu().numpy() > 0.5

    base_tag = f"base_{os.path.splitext(os.path.basename(base_img_path))[0]}"
    donor_tag = f"0_{os.path.splitext(os.path.basename(donor_img_path))[0]}"
    if align_hairline:
        base_npz, donor_npz = ensure_aligned_strands_npz(
            difflocks, base_img_path, donor_img_path, out_path, base_tag, donor_tag)
    else:
        base_npz = ensure_individual_strands_npz(difflocks, base_img_path, out_path, base_tag)
        donor_npz = ensure_individual_strands_npz(difflocks, donor_img_path, out_path, donor_tag)
    base = load_strands_npz(base_npz)
    donor = load_strands_npz(donor_npz)

    print(f"--- joint_contour: per-strand masks of base {os.path.basename(base_img_path)} + "
          f"donor {os.path.basename(donor_img_path)}, then trimming {trim_percent}% ---")
    result = joint_fm_with_trim(
        base["positions"], base["root_uv"], donor["positions"], donor["root_uv"], raw_mask_np,
        ScalpGrid.from_dataset(mask_res),
        root_texels(base["root_uv"], mask_res), root_texels(donor["root_uv"], mask_res),
        trim_percent=trim_percent,
    )
    before, after = result["objectives"]["before"], result["objectives"]["after"]
    t = result["trim_info"]
    print(f"--- joint_contour: (conflict cells, crossing strand pairs) {before} -> {after}; "
          f"trimmed {t['cut_base']} base + {t['cut_donor']} donor strands (max {t['max_cut']}); "
          f"kept base strands {int(result['visible_a'].sum())}/{result['visible_a'].size}, "
          f"kept donor strands {int(result['visible_b'].sum())}/{result['visible_b'].size} ---")

    masks_dir = os.path.join(out_path, "masks")
    save_grayscale_mask(tsai_1.torch.from_numpy(raw_mask_np.astype(np.float32)), masks_dir, "mask_M1_raw")
    save_grayscale_mask(tsai_1.torch.from_numpy(result["binary_mask"].astype(np.float32)), masks_dir,
                        "mask_M1_refined")
    result["initial_state_a"] = raw_mask_np[root_texels(base["root_uv"], mask_res)]
    result["initial_state_b"] = raw_mask_np[root_texels(donor["root_uv"], mask_res)]
    with open(os.path.join(out_path, "trim.json"), "w") as f:
        json.dump(t, f, indent=2)
    return base, donor, result


def run_kung():
    """img2hair_tsai_1.run_kung with refine_mask_per_strand + trimming (--trim_percent)."""
    parser = argparse.ArgumentParser(description="Kung Hair Inference (tsai_2: joint FM + trimming)")
    parser.add_argument("--input_config", type=str, default=None,
                        help="JSON file with 'input_file_paths' (2 images: base, donor) and 'mask_file_paths' (1 mask)")
    parser.add_argument("--blender_path", type=str, default=BLENDER_PATH)
    parser.add_argument("--output_path", type=str, default="./outputs_inference/")
    parser.add_argument("--seed", type=int, default=5,
                        help="diffusion seed, reset before each image (negative = random)")
    parser.add_argument("--align_hairline", action="store_true",
                        help="align the two scalp textures' hairlines (kung's align_hairline_by_shifting) "
                             "before decoding them to strands and fusing")
    parser.add_argument("--trim_percent", type=float, default=TRIM_PERCENT,
                        help="after the joint FM, cut this %% of all strands (base + donor): the visible ones "
                             "crossing the other head the most (0 = no trimming, same result as tsai_1)")
    args = parser.parse_args()
    DiffLocksInference.seed = args.seed if args.seed >= 0 else None

    path_strand_codec = os.path.join(ROOT, "./checkpoints/strand_vae/strand_codec.pt")
    path_config = os.path.join(ROOT, "./configs/config_scalp_texture_conditional.json")
    path_diffusion_model_ckpt_path = os.path.join(ROOT, "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth")
    path_material_model_ckpt_path = os.path.join(ROOT, "./checkpoints/rgb2material/rgb2material.pt")

    out_path = args.output_path
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "masks"), exist_ok=True)

    input_file_paths = ["./samples/hair/freeman_2.png", "./samples/hair/uggams_3.png"]   # base, donor
    mask_file_paths = ["./samples/mask/mask1.png"]
    if args.input_config and os.path.exists(args.input_config):
        with open(args.input_config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        input_file_paths = cfg.get("input_file_paths", input_file_paths)
        mask_file_paths = cfg.get("mask_file_paths", mask_file_paths)
        print(f"Loaded input_config: {args.input_config} "
              f"({len(input_file_paths)} images, {len(mask_file_paths)} masks)")
    elif args.input_config:
        print(f"警告: 找不到 --input_config 指定的檔案 {args.input_config}，使用預設輸入清單。")

    if len(input_file_paths) != 2 or len(mask_file_paths) != 1:
        print(f"Error: this script fuses exactly 2 images (base, donor) with 1 mask, got "
              f"{len(input_file_paths)} images and {len(mask_file_paths)} masks.")
        return

    difflocks = DiffLocksInference(path_strand_codec, path_config, path_diffusion_model_ckpt_path,
                                   path_material_model_ckpt_path)

    # --- 1. reconstruct both hairstyles, per-strand masks (joint FM), trimming ---
    mask_res = difflocks.model_config["input_size"][0]
    base, donor, result = refine_mask_per_strand(
        mask_file_paths[0], input_file_paths[0], input_file_paths[1], difflocks, out_path, mask_res,
        align_hairline=args.align_hairline, trim_percent=args.trim_percent)

    # --- 2a. debug: flux / vector field / grid score analysis ("after" = after trimming) ---
    debug_field_analysis(base, donor, result, out_path)

    # --- 2b. debug: each input's whole hairstyle (kept strands in its color, excluded + trimmed gray) ---
    print("--- debug: rendering each individual input hairstyle ---")
    for img_path, tag, head, keep, color in (
            (input_file_paths[0], "base_", base, result["visible_a"], BASE_STRAND_COLOR),
            (input_file_paths[1], "0_", donor, result["visible_b"], DONOR_STRAND_COLOR)):
        tag += os.path.splitext(os.path.basename(img_path))[0]
        debug_render_individual_hairstyle(
            args.blender_path, out_path, tag, head["positions"],
            np.where(keep[:, None], np.float32(color), np.float32(EXCLUDED_STRAND_COLOR)),
            individual_dir="individual_strands_aligned" if args.align_hairline else "individual_strands")

    # --- 3. fused hair = kept strands after trimming; 4. .blend ---
    npz_out_path = compose_hair_per_strand(base, donor, result, out_path)
    create_blender_file(args, out_path, npz_out_path)


if __name__ == '__main__':
    run_kung()
