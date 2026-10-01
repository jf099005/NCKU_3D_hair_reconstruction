import bpy
import argparse
import os
import shutil
import sys
import tempfile
from mathutils import Vector, Matrix
from PIL import Image, ImageDraw, ImageFont  # bundled with Blender's Python

# Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# RTX 4070 / OptiX
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.refresh_devices()

for device in prefs.devices:
    print(device.type, device.name)
    device.use = (device.type == 'OPTIX')

scene.cycles.device = 'GPU'

# Preview settings
scene.cycles.samples = 64

print("Cycles device:", scene.cycles.device)
print("Samples:", scene.cycles.samples)

# --- Render 5 axis-aligned views (front / back / left / right / up) around the hair + head
# and tile them into ONE image (render_multiview.png) instead of one file per view; the
# bottom-up ("down") view is not rendered. "front" is taken from the scene's existing
# --reference_camera object's current direction/distance to the hair+head (so framing/zoom
# matches what it was already tuned to look like); the other views are derived from it:
# front's horizontal opposite/perpendiculars for back/left/right, world Z+ for up.
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', required=True)
parser.add_argument('--reference_camera', default='Camera', help='existing camera to copy distance/lens/framing from')
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

TARGET_OBJ_NAMES = ["hair_01", "smplx_scalp_blender"]


def compute_target_bbox_center():
    mins, maxs, found = None, None, False
    for name in TARGET_OBJ_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        found = True
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            mins = world_corner.copy() if mins is None else Vector(min(a, b) for a, b in zip(mins, world_corner))
            maxs = world_corner.copy() if maxs is None else Vector(max(a, b) for a, b in zip(maxs, world_corner))
    if not found:
        return Vector((0.0, 0.0, 0.0))
    return (mins + maxs) / 2


def look_at_rotation(cam_loc, target, up_hint):
    """Quaternion for a camera at cam_loc looking at target, with up_hint as the
    approximate "up" direction (camera looks down its local -Z, +Y is up)."""
    forward = (target - cam_loc).normalized()
    right = forward.cross(up_hint)
    if right.length < 1e-6:  # forward ~parallel to up_hint (straight down/up) - pick another hint
        right = forward.cross(Vector((1.0, 0.0, 0.0)))
        if right.length < 1e-6:
            right = forward.cross(Vector((0.0, 1.0, 0.0)))
    right = right.normalized()
    true_up = right.cross(forward).normalized()
    return Matrix((right, true_up, -forward)).transposed().to_quaternion()


target_center = compute_target_bbox_center()
ref_cam = bpy.data.objects.get(args.reference_camera)
if ref_cam is None:
    print(f"WARNING: reference camera '{args.reference_camera}' not found, "
          "falling back to a default front direction/distance")
    distance, front_dir = 1.0, Vector((0.0, -1.0, 0.0))
else:
    front_dir = ref_cam.location - target_center
    distance = front_dir.length
    front_dir.normalize()

world_up = Vector((0.0, 0.0, 1.0))
front_flat = Vector((front_dir.x, front_dir.y, 0.0))
if front_flat.length < 1e-6:  # reference camera happened to be ~straight overhead
    front_flat = Vector((0.0, -1.0, 0.0))
front_flat.normalize()
right_flat = front_flat.cross(world_up).normalized()

VIEWS = {
    "front": front_flat,
    "back": -front_flat,
    "left": -right_flat,
    "right": right_flat,
    "up": world_up,
}
# left -> right order of the tiles in render_multiview.png
TILE_ORDER = ["left", "front", "right", "back", "up"]

cam_data = bpy.data.cameras.new("SixViewCamera")
if ref_cam is not None:
    cam_data.lens = ref_cam.data.lens
    cam_data.sensor_width = ref_cam.data.sensor_width
cam_obj = bpy.data.objects.new("SixViewCamera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
cam_obj.rotation_mode = 'QUATERNION'
scene.camera = cam_obj
scene.render.image_settings.file_format = 'PNG'

os.makedirs(args.out_path, exist_ok=True)
tmp_dir = tempfile.mkdtemp(prefix="gpu_render_", dir=args.out_path)
view_files = {}
for view_name in TILE_ORDER:
    direction = VIEWS[view_name]
    cam_obj.location = target_center + direction * distance
    up_hint = front_flat if abs(direction.dot(world_up)) > 0.99 else world_up
    cam_obj.rotation_quaternion = look_at_rotation(cam_obj.location, target_center, up_hint)
    bpy.context.view_layer.update()

    scene.render.filepath = os.path.join(tmp_dir, f"{view_name}.png")
    bpy.ops.render.render(write_still=True)
    view_files[view_name] = scene.render.filepath

# tile the views side by side (one row, TILE_ORDER), each labeled with its view name
tiles = [Image.open(view_files[v]).convert("RGBA") for v in TILE_ORDER]
w, h = tiles[0].size
label_h = max(24, h // 16)
sheet = Image.new("RGBA", (w * len(tiles), h + label_h), (255, 255, 255, 255))
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default(size=int(label_h * 0.7))
for i, (view_name, tile) in enumerate(zip(TILE_ORDER, tiles)):
    sheet.alpha_composite(tile, (i * w, label_h))
    draw.text((i * w + w // 2, label_h // 2), view_name, fill=(0, 0, 0, 255), font=font, anchor="mm")
out_file = os.path.join(args.out_path, "render_multiview.png")
sheet.convert("RGB").save(out_file)
shutil.rmtree(tmp_dir)
print("wrote render to", out_file)
