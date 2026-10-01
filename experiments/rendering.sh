input_path=${1:-${output_path:-./outputs_inference/}}
output_path=${2:-./render/}
~/blender/blender \
    -b "${input_path%/}/blender_scene.blend" \
    --python gpu_render.py \
    -- \
    --out_path "$(pwd)/${output_path%/}"
