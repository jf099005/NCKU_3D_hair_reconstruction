#!/usr/bin/env python3

import argparse

import torch
from torch.utils.data import DataLoader

from data_loader.dataloader import DiffLocksDataset
from data_loader.mesh_utils import World2Local
from utils.strand_util import compute_dirs


def tensor_stats(name, value):
    value = value.detach()
    finite = torch.isfinite(value)
    if finite.any():
        finite_values = value[finite]
        stats = (
            f"min={finite_values.min().item():.6g}, "
            f"max={finite_values.max().item():.6g}, "
            f"mean={finite_values.mean().item():.6g}"
        )
    else:
        stats = "no finite values"
    print(f"{name}: shape={tuple(value.shape)}, finite={finite.float().mean().item() * 100:.2f}%, {stats}")


def check_finite(name, value):
    if torch.isfinite(value).all():
        return True
    tensor_stats(name, value)
    return False


def main():
    parser = argparse.ArgumentParser(description="Check DiffLocks-format VAE dataset tensors for NaN/Inf after dataloader preprocessing.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--strands_per_hairstyle", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    split_train = None if args.split == "all" else args.split == "train"

    dataset = DiffLocksDataset(
        args.dataset_path,
        train=split_train,
        load_rgb_imgs=False,
        load_full_strands=True,
        load_guide_strands=False,
        load_interpolated_strands=False,
        load_cam=False,
        compute_tbn_full_strands=True,
        nr_full_strands_per_hairstyle=args.strands_per_hairstyle,
        check_validity=True,
        overfit=False,
        train_ratio=args.train_ratio,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    world2local = World2Local()

    bad = 0
    checked = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        sample_name = batch.get("file", ["unknown"])[0]
        full = batch["full_strands"]
        positions = full["positions"].to(device)
        root_normal = full["root_normal"].to(device)
        root_uv = full["root_uv"].to(device)
        tbn = full["tbn"].to(device)

        ok = True
        ok &= check_finite(f"{sample_name}/positions", positions)
        ok &= check_finite(f"{sample_name}/root_normal", root_normal)
        ok &= check_finite(f"{sample_name}/root_uv", root_uv)
        ok &= check_finite(f"{sample_name}/tbn", tbn)

        local_positions, local_normals = world2local(tbn, positions, root_normal)
        local_positions = local_positions.reshape(-1, 256, 3)
        dirs = compute_dirs(local_positions, append_last_dir=False)
        curv = compute_dirs(dirs, append_last_dir=False)

        ok &= check_finite(f"{sample_name}/local_positions", local_positions)
        ok &= check_finite(f"{sample_name}/local_normals", local_normals)
        ok &= check_finite(f"{sample_name}/dirs", dirs)
        ok &= check_finite(f"{sample_name}/curv", curv)

        checked += 1
        if not ok:
            bad += 1
            print(f"BAD sample: {sample_name}")
        elif checked <= 3:
            print(f"OK sample: {sample_name}")

    print(f"Checked {checked} samples, bad samples: {bad}")


if __name__ == "__main__":
    main()
