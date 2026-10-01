# Runs inside Blender's Python (bpy) - not a standalone script.
#
# Loads reconstructed 3D hair strands (positions npz from inference_difflocks.py,
# e.g. difflocks_output_strands.npz) into this repo's own head+hair visualization
# scene/asset (./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend,
# which already contains the head/body mesh "smplx_base_blender" alongside the
# hair curves object "hair_01") and renders several preset camera angles of it
# (front / three-quarter / side - the scene already ships cameras for these).
#
# By default colors the hair using DiffLocks' own predicted melanin/melanin_redness
# (read from hair.json next to the input npz, written by inference_difflocks.py),
# wired into the scene's Principled Hair BSDF MELANIN parametrization - the same
# mechanism as hair_color_reconstruction/render_frontview_blender.py. Without this,
# the scene's hair material defaults to a per-strand random rainbow debug tint,
# not an actual hair color.
#
# Alternatively --hair_rgb / --hair_rgb_json colors the hair with one flat RGB
# color (e.g. the median-color method of img2hair_kung_mul2.py, see
# extract_median_hair_color.py), using the same Principled BSDF + VertexColor
# attribute setup as npz2blender_kung.py.
#
# Geometry-loading logic mirrors ../inference/npz2blender.py and
# ../hair_color_reconstruction/render_frontview_blender.py exactly.
#
# Usage:
#   <blender> -t <threads> --background --python ./inference/render_multiview_blender.py -- \
#       --input_npz <difflocks_output_strands.npz> --out_dir <output_dir> \
#       [--hair_json <hair.json>] [--melanin_amount M --melanin_redness R] \
#       [--hair_rgb R,G,B | --hair_rgb_json <hair_color_median.json>] \
#       [--views front,three_quarter,side] [--samples N] [--resolution N] \
#       [--strands_subsample F] [--shrinkwrap]

import bpy
import numpy as np
import argparse
import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_BLEND = os.path.join(SCRIPT_DIR, "assets", "blender_vis_base_v26_with_shrinkwrap_full_base.blend")

VIEW_TO_CAMERA = {
    "front": "Camera",
    "three_quarter": "Camera.002",
    "side": "Camera_side",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--hair_json', default=None, help='hair.json with predicted melanin/redness (defaults to hair.json next to --input_npz)')
    parser.add_argument('--melanin_amount', type=float, default=None, help='overrides the value from hair.json')
    parser.add_argument('--melanin_redness', type=float, default=None, help='overrides the value from hair.json')
    parser.add_argument('--hair_rgb', default=None, help='flat hair color "r,g,b" in [0,1]; replaces the melanin coloring')
    parser.add_argument('--hair_rgb_json', default=None, help='json with an "rgb" field (e.g. hair_color_median.json); replaces the melanin coloring')
    parser.add_argument('--seed', type=int, default=0, help='seed for strand subsampling, so different colorings render the same strands')
    parser.add_argument('--views', default="front,three_quarter,side", help=f'comma-separated subset of {list(VIEW_TO_CAMERA)}')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--strands_subsample', type=float, default=0.5, help='fraction of strands to keep, for a faster render')
    parser.add_argument('--shrinkwrap', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    hair_geom = np.load(args.input_npz)
    points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3

    if args.strands_subsample != 1.0:
        np.random.seed(args.seed)
        num_keep = int(points.shape[0] * args.strands_subsample)
        keep_idx = np.random.choice(points.shape[0], num_keep, replace=False)
        points = points[keep_idx, :, :].copy()

    bpy.ops.wm.open_mainfile(filepath=BASE_BLEND)

    obj = bpy.data.objects.get("hair_01")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    curves_data = obj.data

    nr_strands = points.shape[0]
    nr_points_per_strand = points.shape[1]
    points_per_curve = [nr_points_per_strand for _ in range(nr_strands)]
    curves_data.add_curves(points_per_curve)

    flat_points = points.reshape(-1, 3)
    flat_points[:, [1, 2]] = flat_points[:, [2, 1]]  # match the coordinate convention used in npz2blender.py
    flat_points[:, 1] *= -1
    curves_data.points.foreach_set("position", flat_points.flatten())

    if not args.shrinkwrap:
        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")

    obj.data.update_tag()
    obj.modifiers.update()
    bpy.context.view_layer.update()

    hair_rgb = None
    if args.hair_rgb is not None:
        hair_rgb = [float(c) for c in args.hair_rgb.split(",")]
    elif args.hair_rgb_json is not None:
        with open(args.hair_rgb_json) as f:
            hair_rgb = json.load(f)["rgb"]

    if hair_rgb is not None:
        apply_flat_rgb_coloring(obj, hair_rgb)
    else:
        apply_melanin_coloring(args)

    scene = bpy.context.scene
    scene.cycles.samples = args.samples
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = 'PNG'

    os.makedirs(args.out_dir, exist_ok=True)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for view in views:
        cam_name = VIEW_TO_CAMERA.get(view)
        if cam_name is None:
            print(f"WARNING: unknown view '{view}', skipping (known: {list(VIEW_TO_CAMERA)})")
            continue
        cam_obj = bpy.data.objects.get(cam_name)
        if cam_obj is None:
            print(f"WARNING: camera '{cam_name}' not found in scene, skipping view '{view}'")
            continue
        scene.camera = cam_obj
        out_path = os.path.join(args.out_dir, f"render_{view}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print("wrote render to", out_path)


def apply_melanin_coloring(args):
    # resolve predicted melanin/redness: explicit args > hair.json > none (leaves debug material untouched)
    melanin_amount = args.melanin_amount
    melanin_redness = args.melanin_redness
    if melanin_amount is None or melanin_redness is None:
        hair_json_path = args.hair_json or os.path.join(os.path.dirname(os.path.abspath(args.input_npz)), "hair.json")
        if os.path.isfile(hair_json_path):
            with open(hair_json_path) as f:
                hair_data = json.load(f)
            melanin_amount = hair_data["melanin"] if melanin_amount is None else melanin_amount
            melanin_redness = hair_data["redness"] if melanin_redness is None else melanin_redness
            print(f"using predicted hair color from {hair_json_path}: melanin={melanin_amount} redness={melanin_redness}")
        else:
            print(f"WARNING: no hair.json found at {hair_json_path} and no --melanin_amount/--melanin_redness given - "
                  f"rendering with the scene's default debug rainbow tint, NOT the predicted hair color")

    if melanin_amount is not None and melanin_redness is not None:
        mat = bpy.data.materials.get("Bgen_Hair_Shader")
        bsdf = mat.node_tree.nodes.get("Cycles bsdf.001")
        bsdf.parametrization = 'MELANIN'
        bsdf.inputs["Melanin"].default_value = melanin_amount
        bsdf.inputs["Melanin Redness"].default_value = melanin_redness


def apply_flat_rgb_coloring(obj, rgb):
    # same as npz2blender_kung.py: per-point "VertexColor" attribute -> Principled BSDF Base Color.
    # The rgb values are written as-is, like npz2blender_kung.py does with the npz color_map.
    print(f"using flat hair color rgb={rgb}")
    curves_data = obj.data
    attr_name = "VertexColor"
    color_attr = curves_data.attributes.get(attr_name) or \
                 curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
    full_colors = np.ones((len(curves_data.points), 4))
    full_colors[:, :3] = rgb
    color_attr.data.foreach_set("color", full_colors.flatten())

    mat = bpy.data.materials.get("Hair_Mask_Material") or bpy.data.materials.new(name="Hair_Mask_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    node_attr = nodes.new(type='ShaderNodeAttribute')
    node_attr.attribute_name = attr_name
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.inputs['Roughness'].default_value = 0.7
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


if __name__ == '__main__':
    main()
