#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader.dataloader import DiffLocksDataset
from data_loader.mesh_utils import World2Local
from models.strand_codec import StrandCodec
from utils.strand_util import compute_dirs


class EvalHyperParams:
    def __init__(self):
        self.normalize_input = True
        self.decode_type = "dir"
        self.scale_init = 30.0
        self.nr_verts_per_strand = 256
        self.nr_values_to_decode = 255
        self.dim_per_value_decoded = 3
        self.latent_dim = 64
        self.modulation_hidden_dim = 128
        self.siren_hidden_dim = 128


def prepare_gt_batch(batch, device):
    tbn = batch["full_strands"]["tbn"].to(device)
    positions = batch["full_strands"]["positions"].to(device)
    root_normal = batch["full_strands"]["root_normal"].to(device)

    strand_positions, _ = World2Local()(tbn, positions, root_normal)
    strand_positions = strand_positions.reshape(-1, 256, 3)
    strand_directions = compute_dirs(strand_positions, append_last_dir=False)

    return {
        "strand_positions": strand_positions,
        "strand_directions": strand_directions,
    }


def local_to_world(local_positions, batch, device):
    tbn = batch["full_strands"]["tbn"].to(device)
    world_positions = batch["full_strands"]["positions"].to(device)
    root_pos = world_positions[:, :, 0:1, :]

    indices_tbn = torch.tensor([0, 2, 1], device=device).long()
    tbn = torch.index_select(tbn, 3, indices_tbn)
    tbn[..., 0] = -tbn[..., 0]

    batch_size, nr_strands = tbn.shape[:2]
    local_positions = local_positions.reshape(batch_size, nr_strands, -1, 3, 1)
    tbn = tbn.reshape(batch_size, nr_strands, 1, 3, 3)
    out = torch.matmul(tbn, local_positions).reshape(batch_size, nr_strands, -1, 3)
    return out + root_pos


def load_normalization(path, device):
    normalization_dict = torch.load(path, map_location=device)
    return {key: value.to(device) for key, value in normalization_dict.items()}


def default_normalization_path(checkpoint_path):
    models_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    checkpoint_epoch_dir = os.path.dirname(models_dir)
    experiment_dir = os.path.dirname(checkpoint_epoch_dir)
    path = os.path.join(experiment_dir, "normalization_data.pt")
    return path if os.path.isfile(path) else None


def load_hyperparams(checkpoint_path):
    models_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    hyperparams_path = os.path.join(models_dir, "hyperparams.json")
    hyperparams = EvalHyperParams()
    if not os.path.isfile(hyperparams_path):
        return hyperparams

    with open(hyperparams_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    for key, value in saved.items():
        if hasattr(hyperparams, key):
            setattr(hyperparams, key, value)
    return hyperparams


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained StrandVAE checkpoint on a DiffLocks-format dataset.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to strand_codec.pt")
    parser.add_argument("--out_path", default="./outputs_vae_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--strands_per_hairstyle", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--normalization_path", default=None, help="Path to normalization_data.pt saved by train_strandsVAE.py")
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
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    hyperparams = load_hyperparams(args.checkpoint)
    model = StrandCodec(
        do_vae=True,
        scale_init=hyperparams.scale_init,
        nr_verts_per_strand=hyperparams.nr_verts_per_strand,
        nr_values_to_decode=hyperparams.nr_values_to_decode,
        dim_per_value_decoded=hyperparams.dim_per_value_decoded,
        latent_dim=hyperparams.latent_dim,
        modulation_hidden_dim=hyperparams.modulation_hidden_dim,
        siren_hidden_dim=hyperparams.siren_hidden_dim,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing checkpoint keys initialized from current model:", missing_keys)
    if unexpected_keys:
        print("Unexpected checkpoint keys ignored:", unexpected_keys)
    model.eval()
    model.encoder.do_vae = False

    normalization_path = args.normalization_path or default_normalization_path(args.checkpoint)
    if normalization_path is not None:
        print(f"Loading normalization stats from {normalization_path}")
        normalization_dict = load_normalization(normalization_path, device)
    else:
        print("No normalization_data.pt found; falling back to original DiffLocks normalization stats.")
        normalization_dict = dataset.get_normalization_data(device)
    os.makedirs(args.out_path, exist_ok=True)

    total_l1 = 0.0
    total_batches = 0
    saved_files = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            gt_dict = prepare_gt_batch(batch, device)
            pred_dict, latent_dict = model(gt_dict, hyperparams, normalization_dict)

            gt_local = gt_dict["strand_positions"]
            pred_local = pred_dict["strand_positions"]
            l1 = torch.mean(torch.abs(gt_local - pred_local)).item()
            total_l1 += l1
            total_batches += 1

            gt_world = local_to_world(gt_local, batch, device)
            pred_world = local_to_world(pred_local, batch, device)

            file_stem = f"batch_{batch_idx:04d}"
            sample_names = [str(name) for name in batch.get("file", [])]
            gt_path = os.path.join(args.out_path, f"{file_stem}_gt_strands.npz")
            pred_path = os.path.join(args.out_path, f"{file_stem}_pred_strands.npz")
            gt_local_path = os.path.join(args.out_path, f"{file_stem}_gt_local_strands.npz")
            pred_local_path = os.path.join(args.out_path, f"{file_stem}_pred_local_strands.npz")
            local_path = os.path.join(args.out_path, f"{file_stem}_local_compare.npz")
            files_path = os.path.join(args.out_path, f"{file_stem}_files.txt")

            np.savez(gt_path, positions=gt_world.reshape(-1, 256, 3).cpu().numpy())
            np.savez(pred_path, positions=pred_world.reshape(-1, 256, 3).cpu().numpy())
            np.savez(gt_local_path, positions=gt_local.cpu().numpy())
            np.savez(pred_local_path, positions=pred_local.cpu().numpy())
            np.savez(
                local_path,
                gt_positions=gt_local.cpu().numpy(),
                pred_positions=pred_local.cpu().numpy(),
                z=latent_dict["z"].cpu().numpy(),
                l1=np.array([l1], dtype=np.float32),
                files=np.array(sample_names),
            )
            with open(files_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sample_names))
                f.write("\n")
            saved_files.extend([gt_path, pred_path, gt_local_path, pred_local_path, local_path, files_path])
            print(f"{file_stem}: local L1 = {l1:.6f}, files = {sample_names}")

    if total_batches == 0:
        raise RuntimeError("No batches were evaluated.")

    print(f"Average local L1: {total_l1 / total_batches:.6f}")
    print(f"Wrote {len(saved_files)} files to {args.out_path}")


if __name__ == "__main__":
    main()
