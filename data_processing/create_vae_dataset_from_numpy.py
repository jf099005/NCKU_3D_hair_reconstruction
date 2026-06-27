#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.path.pardir))
DEFAULT_BODY_DATA = os.path.join(REPO_ROOT, "data_loader", "difflocks_bodydata")


def load_strand_data(path, key):
    extra = {}
    if path.endswith(".npy"):
        positions = np.load(path)
    else:
        data = np.load(path)
        positions = data[key]
        for extra_key in ("root_uv", "root_normal", "tbn"):
            if extra_key in data:
                extra[extra_key] = np.asarray(data[extra_key], dtype=np.float32)

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("Expected strand positions with shape (n_strands, n_points, 3)")
    return positions, extra


def resample_strands(positions, target_points):
    n_strands, n_points, _ = positions.shape
    if n_points == target_points:
        return positions.astype(np.float32, copy=False)

    src_t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_points, dtype=np.float32)
    out = np.empty((n_strands, target_points, 3), dtype=np.float32)

    for strand_idx in range(n_strands):
        for axis in range(3):
            out[strand_idx, :, axis] = np.interp(dst_t, src_t, positions[strand_idx, :, axis])

    return out


def estimate_root_normals(positions):
    if positions.shape[1] < 2:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (positions.shape[0], 1))

    root_normal = positions[:, 1, :] - positions[:, 0, :]
    lengths = np.linalg.norm(root_normal, axis=1, keepdims=True)
    safe = lengths[:, 0] > 1e-8
    root_normal[~safe] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    root_normal[safe] = root_normal[safe] / lengths[safe]
    return root_normal.astype(np.float32)


def default_root_uv(positions):
    roots = positions[:, 0, :]
    mins = roots[:, [0, 1]].min(axis=0, keepdims=True)
    maxs = roots[:, [0, 1]].max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-8)
    return ((roots[:, [0, 1]] - mins) / denom).astype(np.float32)


def identity_tbn(n_strands):
    return np.tile(np.eye(3, dtype=np.float32)[None, :, :], (n_strands, 1, 1))


def root_relative_positions(positions):
    return positions - positions[:, 0:1, :]


def write_chunks(positions, root_uv, root_normal, tbn, out_dir, chunk_sizes):
    if isinstance(chunk_sizes, int):
        chunk_sizes = [chunk_sizes]
    for chunk_size in chunk_sizes:
        write_chunk_size(positions, root_uv, root_normal, tbn, out_dir, chunk_size)


def write_chunk_size(positions, root_uv, root_normal, tbn, out_dir, chunk_size):
    chunk_dir = os.path.join(out_dir, "full_strands_chunked", f"nr_strands_{chunk_size}")
    os.makedirs(chunk_dir, exist_ok=True)

    n_chunks = max(1, math.ceil(positions.shape[0] / chunk_size))
    for chunk_idx, strand_indices in enumerate(np.array_split(np.arange(positions.shape[0]), n_chunks)):
        np.savez(
            os.path.join(chunk_dir, f"{chunk_idx}.npz"),
            positions=positions[strand_indices],
            root_uv=root_uv[strand_indices],
            root_normal=root_normal[strand_indices],
            tbn=tbn[strand_indices],
        )


def convert_one(input_path, out_path, sample_name, npz_key, target_points, chunk_size, root_relative):
    positions, extra = load_strand_data(input_path, npz_key)
    positions = resample_strands(positions, target_points)
    root_uv = extra.get("root_uv", default_root_uv(positions))
    if root_relative:
        positions = root_relative_positions(positions)
    root_normal = extra.get("root_normal", estimate_root_normals(positions))
    tbn = extra.get("tbn", identity_tbn(positions.shape[0]))

    hairstyle_dir = os.path.join(out_path, "generated_hairstyles", sample_name)
    os.makedirs(hairstyle_dir, exist_ok=True)

    np.savez(
        os.path.join(hairstyle_dir, "full_strands.npz"),
        positions=positions,
        root_uv=root_uv,
        root_normal=root_normal,
        tbn=tbn,
    )
    write_chunks(positions, root_uv, root_normal, tbn, hairstyle_dir, chunk_size)

    with open(os.path.join(hairstyle_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"mirror_hair": False}, f, indent=2)

    np.savez(os.path.join(hairstyle_dir, "numpy_state.npz"), done=np.array([1], dtype=np.uint8))

    print(f"Wrote {sample_name}: positions {positions.shape}")


def copy_body_data(out_path, body_data):
    body_out = os.path.join(out_path, "body_data")
    os.makedirs(body_out, exist_ok=True)
    for filename in ("scalp.ply", "smplx_base.ply"):
        shutil.copy2(os.path.join(body_data, filename), os.path.join(body_out, filename))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a raw strand numpy array to the minimal DiffLocks dataset layout used by train_strandsVAE.py."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to one .npy or .npz containing strand positions")
    input_group.add_argument("--input_root", help="Folder containing sample subfolders, each with hair_data.npy")
    parser.add_argument("--out_path", required=True, help="Output dataset root")
    parser.add_argument("--sample_name", default="custom_000", help="Name of the generated hairstyle folder")
    parser.add_argument("--hair_filename", default="hair_data.npy", help="Filename to read under each --input_root sample folder")
    parser.add_argument("--npz_key", default="positions", help="Key to read when --input is .npz")
    parser.add_argument("--target_points", type=int, default=256, help="Number of points per strand expected by the VAE")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size used by train_strandsVAE.py")
    parser.add_argument("--extra_chunk_size", type=int, action="append", default=[1000], help="Additional chunk size to write. Can be passed multiple times.")
    parser.add_argument("--body_data", default=DEFAULT_BODY_DATA, help="Folder containing scalp.ply and smplx_base.ply")
    parser.add_argument(
        "--keep_world_coords",
        action="store_true",
        help="Keep raw world coordinates. By default strands are made root-relative, which is safer for custom VAE training.",
    )
    args = parser.parse_args()
    root_relative = not args.keep_world_coords
    chunk_sizes = sorted(set([args.chunk_size] + args.extra_chunk_size))

    if args.input:
        convert_one(args.input, args.out_path, args.sample_name, args.npz_key, args.target_points, chunk_sizes, root_relative)
    else:
        sample_dirs = [
            os.path.join(args.input_root, name)
            for name in sorted(os.listdir(args.input_root))
            if os.path.isdir(os.path.join(args.input_root, name))
        ]
        if not sample_dirs:
            raise ValueError(f"No sample folders found under {args.input_root}")

        converted = 0
        for sample_dir in sample_dirs:
            input_path = os.path.join(sample_dir, args.hair_filename)
            if not os.path.isfile(input_path):
                print(f"Skipping {sample_dir}: missing {args.hair_filename}")
                continue
            convert_one(
                input_path,
                args.out_path,
                os.path.basename(sample_dir),
                args.npz_key,
                args.target_points,
                chunk_sizes,
                root_relative,
            )
            converted += 1
        print(f"Converted {converted} samples")

    copy_body_data(args.out_path, args.body_data)


if __name__ == "__main__":
    main()
