input_path=${1:-${output_path:-./outputs_inference/}}
output_path=${2:-./render/}
~/blender-5.2.0-linux-x64/blender \
    -b "${input_path%/}/blender_scene.blend" \
    --python gpu_render.py \
    -o "$(pwd)/${output_path%/}/test" \
    -F PNG \
    -f 1
