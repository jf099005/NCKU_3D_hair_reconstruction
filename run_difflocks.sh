#!/bin/bash
# Hair-color comparison on one image. DiffLocks inference runs once (shared
# strand geometry), then the same strands are rendered multi-view twice:
#   1) renders_difflocks_color/ : DiffLocks' own rgb2material melanin/redness (hair.json)
#   2) renders_median_color/    : flat median hair-pixel RGB, as in inference/img2hair_kung_mul2.py
# plus comparison_<view>.png with the two side by side (left: DiffLocks, right: median).
# Usage: ./run_difflocks.sh <img_path> [out_path] [blender_path]
set -e

IMG_PATH="$1"
OUT_PATH="${2:-./outputs_inference/}"
BLENDER_PATH="${3:-/home/kyh/blender/blender}"

if [ -z "$IMG_PATH" ]; then
    echo "Usage: $0 <img_path> [out_path] [blender_path]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT_DIR="${OUT_PATH%/}"
mkdir -p "$OUT_DIR"

python inference_difflocks.py \
    --img_path="$IMG_PATH" \
    --out_path="$OUT_PATH"

python ./inference/extract_median_hair_color.py \
    --img_path="$IMG_PATH" \
    --out_path="$OUT_DIR"

# 1) DiffLocks built-in hair color (melanin/redness from hair.json)
"$BLENDER_PATH" -t 8 --background --python ./inference/render_multiview_blender.py -- \
    --input_npz="$OUT_DIR/difflocks_output_strands.npz" \
    --hair_json="$OUT_DIR/hair.json" \
    --out_dir="$OUT_DIR/renders_difflocks_color"

# 2) median-based hair color (img2hair_kung_mul2.py method)
"$BLENDER_PATH" -t 8 --background --python ./inference/render_multiview_blender.py -- \
    --input_npz="$OUT_DIR/difflocks_output_strands.npz" \
    --hair_rgb_json="$OUT_DIR/hair_color_median.json" \
    --out_dir="$OUT_DIR/renders_median_color"

python - "$OUT_DIR" <<'EOF'
import os, sys, cv2, numpy as np
out = sys.argv[1]
for view in ["front", "three_quarter", "side"]:
    a = cv2.imread(os.path.join(out, "renders_difflocks_color", f"render_{view}.png"))
    b = cv2.imread(os.path.join(out, "renders_median_color", f"render_{view}.png"))
    if a is None or b is None:
        continue
    cv2.imwrite(os.path.join(out, f"comparison_{view}.png"), np.concatenate([a, b], axis=1))
EOF

echo "DiffLocks-color renders: $OUT_DIR/renders_difflocks_color/"
echo "Median-color renders:    $OUT_DIR/renders_median_color/"
echo "Side-by-side:            $OUT_DIR/comparison_*.png (left: DiffLocks, right: median)"
