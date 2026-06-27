#!/usr/bin/env python3

import argparse
import os

import numpy as np


def load_positions(path, key):
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} does not contain key '{key}'. Available keys: {list(data.keys())}")
    positions = np.asarray(data[key])
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (n_strands, n_points, 3), got {positions.shape}")
    return positions


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.5)
    if radius <= 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_strands(ax, positions, view, linewidth, alpha):
    strand_colors = positions[:, -1, 2] - positions[:, 0, 2]
    strand_colors = (strand_colors - strand_colors.min()) / (np.ptp(strand_colors) + 1e-8)

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    for strand, color_value in zip(positions, strand_colors):
        ax.plot(
            strand[:, 0],
            strand[:, 1],
            strand[:, 2],
            color=cmap(float(color_value)),
            linewidth=linewidth,
            alpha=alpha,
        )

    flat = positions.reshape(-1, 3)
    set_axes_equal(ax, flat)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)


def plot_strands(positions, out_path, max_strands, view, linewidth, alpha, multi_view):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_strands = positions.shape[0]
    if max_strands is not None and positions.shape[0] > max_strands:
        indices = np.linspace(0, positions.shape[0] - 1, max_strands).astype(np.int64)
        positions = positions[indices]
    plotted_strands = positions.shape[0]

    if multi_view:
        views = [(20.0, -70.0), (15.0, 0.0), (80.0, -90.0)]
        titles = ["Perspective", "Side", "Top"]
        fig = plt.figure(figsize=(18, 6), dpi=180)
        for idx, (cur_view, title) in enumerate(zip(views, titles), start=1):
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            draw_strands(ax, positions, cur_view, linewidth, alpha)
            ax.set_title(title)
    else:
        fig = plt.figure(figsize=(8, 8), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
        draw_strands(ax, positions, view, linewidth, alpha)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, transparent=False)
    plt.close(fig)
    return total_strands, plotted_strands


def main():
    parser = argparse.ArgumentParser(description="Plot a strand .npz file containing positions with shape (n_strands, n_points, 3).")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--key", default="positions")
    parser.add_argument("--max_strands", type=int, default=1000)
    parser.add_argument("--view", type=float, nargs=2, default=(20.0, -70.0), metavar=("ELEV", "AZIM"))
    parser.add_argument("--linewidth", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--multi_view", action="store_true", help="Render perspective, side, and top views in one image.")
    args = parser.parse_args()

    positions = load_positions(args.input_npz, args.key)
    total_strands, plotted_strands = plot_strands(positions, args.out_png, args.max_strands, args.view, args.linewidth, args.alpha, args.multi_view)
    print(f"Input strands: {total_strands}; plotted strands: {plotted_strands}")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
