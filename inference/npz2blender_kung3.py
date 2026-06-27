#run with 
# ~/blender-4.1.1-linux-x64/blender  -t 8 --background --python ./inference/npz2blender.py -- --input_npz <NPZ_PATH> --out_path <OUTPUT_PATH> --export_alembic

# 頭髮老化

import bpy
from bpy.app.handlers import persistent
import bpy_extras
import os
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join

import math
from mathutils import Matrix, Vector
import mathutils
import shutil
import time

from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torchvision import transforms

path_cur_script = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================
# 【全新補上】利用被註解的舊 main 核心邏輯包裝而成的建立單影格 Mesh 函數
# =====================================================================================
def create_hair_frame_mesh(npz_path, frame_idx, args):
    """
    讀取單一 npz 檔案，在 Blender 中建立對應的髮絲物件並上色
    """
    if not os.path.exists(npz_path):
        print(f"警告：找不到該格的 npz 檔案: {npz_path}")
        return None

    # 1. 讀取 npz 幾何與顏色
    hair_geom = np.load(npz_path)
    points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3
    strand_colors_raw = hair_geom.get("colors")

    # 2. 隨機抽樣 (Subsample) 邏輯（直接搬移自你原本被註解的 code）
    if args.strands_subsample != 1.0 or args.vertex_subsample != 1.0:
        num_strands_to_keep = int(points.shape[0] * args.strands_subsample)
        strands_to_keep = np.random.choice(points.shape[0], num_strands_to_keep, replace=False)
        points = points[strands_to_keep, :, :].copy()

        nr_verts_to_skip = int(np.floor(1.0 / args.vertex_subsample))
        points = points[:, ::nr_verts_to_skip, :].copy()

        if strand_colors_raw is not None:
            strand_colors_raw = strand_colors_raw[strands_to_keep]

    nr_strands = points.shape[0]
    nr_points_per_strand = points.shape[1]

    # 3. 獲取場景裡的 hair_01 基底模板
    base_obj = bpy.data.objects.get("hair_01")
    if not base_obj:
        print("錯誤：基礎場景中找不到 'hair_01' 物件！")
        return None

    # 複製一個新物件給當前影格（避免多個影格互相污染）
    new_obj = base_obj.copy()
    new_obj.data = base_obj.data.copy()
    new_obj.name = f"Hair_Frame_{frame_idx:03d}"
    bpy.context.collection.objects.link(new_obj)

    curves_data = new_obj.data

    # 4. 寫入 3D 頂點坐標
    points_per_curve = [nr_points_per_strand for _ in range(nr_strands)]
    curves_data.add_curves(points_per_curve)

    # 調整 Blender 的座標軸軸向 (Swap Y and Z, Negate Y)
    flat_points = points.reshape(-1, 3)
    flat_points[:, [1, 2]] = flat_points[:, [2, 1]]  
    flat_points[:, 1] *= -1  

    curves_data.points.foreach_set("position", flat_points.flatten())

    # 5. 寫入頂點顏色
    attr_name = "VertexColor"
    if strand_colors_raw is not None:
        if len(strand_colors_raw) != nr_strands:
            strand_colors_raw = strand_colors_raw[:nr_strands]
        
        color_attr = curves_data.attributes.get(attr_name) or \
                     curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

        num_total_points = nr_strands * nr_points_per_strand
        full_colors_rgba = np.ones((num_total_points, 4))
        
        repeated_rgb = np.repeat(strand_colors_raw, nr_points_per_strand, axis=0)
        full_colors_rgba[:, :3] = repeated_rgb

        color_attr.data.foreach_set("color", full_colors_rgba.flatten())

    # 6. 處理 Shrinkwrap 修正
    if not args.shrinkwrap:
        for mod in list(new_obj.modifiers):
            if mod.name == "Shrinkwrap Hair Curves":
                new_obj.modifiers.remove(mod)

    # 更新物件狀態
    new_obj.data.update_tag()
    new_obj.modifiers.update()
    
    # 連結並確保材質啟用
    setup_hair_material_nodes(new_obj.name, attr_name=attr_name)
    
    return new_obj


def update_hair_single_object(npz_path, args):
    """
    讀取單一 npz 檔案，直接更新場景中唯一的 'hair_01' 幾何與頂點屬性，
    不建立新物件，以此完美保留幾何節點的髮型流向，並額外寫入毛髮半徑控制。
    """
    if not os.path.exists(npz_path):
        print(f"警告：找不到該格的 npz 檔案: {npz_path}")
        return False

    # 1. 讀取 npz 幾何、顏色與密度
    hair_geom = np.load(npz_path)
    points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3
    strand_colors_raw = hair_geom.get("colors")
    strand_densities_raw = hair_geom.get("densities")  # 💡 讀取第一步存在 Python 的密度數據

    # 2. 隨機抽樣邏輯 (維持你原本的規格)
    if args.strands_subsample != 1.0 or args.vertex_subsample != 1.0:
        num_strands_to_keep = int(points.shape[0] * args.strands_subsample)
        strands_to_keep = np.random.choice(points.shape[0], num_strands_to_keep, replace=False)
        points = points[strands_to_keep, :, :].copy()

        nr_verts_to_skip = int(np.floor(1.0 / args.vertex_subsample))
        points = points[:, ::nr_verts_to_skip, :].copy()

        if strand_colors_raw is not None:
            strand_colors_raw = strand_colors_raw[strands_to_keep]
        if strand_densities_raw is not None:
            strand_densities_raw = strand_densities_raw[strands_to_keep]

    nr_strands = points.shape[0]
    nr_points_per_strand = points.shape[1]

    # 3. 💡【核心改動】獲取場景中唯一的主頭髮物件，不進行複製
    main_obj = bpy.data.objects.get("hair_01")
    if not main_obj:
        print("錯誤：基礎場景中找不到 'hair_01' 物件！")
        return False

    curves_data = main_obj.data
    curves_data.clear()  # 乾淨清空上一格的舊網格

    # 4. 重新寫入 3D 頂點坐標 (此時為 100% 完整自然的導引線形狀)
    points_per_curve = [nr_points_per_strand for _ in range(nr_strands)]
    curves_data.add_curves(points_per_curve)

    # 調整座標軸向
    flat_points = points.reshape(-1, 3)
    flat_points[:, [1, 2]] = flat_points[:, [2, 1]]  
    flat_points[:, 1] *= -1  
    curves_data.points.foreach_set("position", flat_points.flatten())

    # 5. 寫入頂點顏色
    attr_name = "VertexColor"
    if strand_colors_raw is not None:
        if len(strand_colors_raw) != nr_strands:
            strand_colors_raw = strand_colors_raw[:nr_strands]
        
        color_attr = curves_data.attributes.get(attr_name) or \
                     curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

        num_total_points = nr_strands * nr_points_per_strand
        full_colors_rgba = np.ones((num_total_points, 4))
        repeated_rgb = np.repeat(strand_colors_raw, nr_points_per_strand, axis=0)
        full_colors_rgba[:, :3] = repeated_rgb
        color_attr.data.foreach_set("color", full_colors_rgba.flatten())

    # 6. 💡【全新第二步：寫入毛髮半徑比例屬性】
    if strand_densities_raw is not None:
        attr_density_name = "Hair_Radius_Scale"  # 幾何節點專用的具名屬性
        
        # 建立 FLOAT 類型的頂點屬性
        density_attr = curves_data.attributes.get(attr_density_name) or \
                       curves_data.attributes.new(name=attr_density_name, type='FLOAT', domain='POINT')
        
        # 把每根頭髮的密度擴展到所有頂點
        full_densities = np.repeat(strand_densities_raw, nr_points_per_strand, axis=0)
        density_attr.data.foreach_set("value", full_densities.flatten())

    # 7. 處理 Shrinkwrap 修正 (維持你原本的邏輯)
    if not args.shrinkwrap:
        for mod in list(main_obj.modifiers):
            if mod.name == "Shrinkwrap Hair Curves":
                main_obj.modifiers.remove(mod)

    # 通知 Blender 更新數據圖
    main_obj.data.update_tag()
    main_obj.modifiers.update()
    
    # 確保材質與唯一的物件綁定
    setup_hair_material_nodes(main_obj.name, attr_name=attr_name)
    
    return True


def export_alembic(out_alembic_path, resolution):update_hair_for_frame
    print("-------------------------------------------")
    bpy.data.objects["hair_01"].select_set(True)
    hair = bpy.data.objects["hair_01"]
    bpy.context.view_layer.objects.active = hair

    start = time.time()
    for modif in hair.modifiers:
        print("applying", modif.name)
        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.modifier_apply(modifier=modif.name)
    print("finished applying all geometry nodes")
    end = time.time()
    print("applying geometry nodes took", end - start)

    bpy.ops.curves.convert_to_particle_system()
    bpy.context.object.show_instancer_for_render = False
    bpy.context.object.show_instancer_for_viewport = False
    
    bpy.data.particles["ParticleSettings"].display_step = resolution
    bpy.data.particles["ParticleSettings"].hair_step = resolution
    bpy.data.particles["ParticleSettings"].render_step = resolution
    
    for obj in bpy.data.objects:
        if obj.name != "smplx_scalp_blender":
            obj.hide_render = True
            obj.hide_viewport = True

    bpy.data.objects['smplx_scalp_blender'].hide_render = False
    bpy.data.objects['smplx_scalp_blender'].hide_viewport = False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_render = False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_viewport = False
    bpy.data.objects["smplx_scalp_blender"].select_set(True)

    bpy.ops.wm.alembic_export(filepath=out_alembic_path, check_existing=False, start=1, end=1, selected=True, visible_objects_only=True, uvs=False, packuv=False, normals=False, use_instancing=False, global_scale=1.0, export_hair=True, export_particles=False, as_background_job=False, evaluation_mode='VIEWPORT', init_scene_frame_range=True)


def setup_hair_material_nodes(obj_name, attr_name="VertexColor"):
    obj = bpy.data.objects.get(obj_name)
    if not obj: return

    mat_name = "Hair_Final_Material"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_attr = nodes.new(type='ShaderNodeAttribute')
    node_attr.attribute_name = attr_name
    
    node_hair = nodes.new(type='ShaderNodeBsdfHairPrincipled')
    node_hair.location = (0, 300)
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 300)

    links.new(node_attr.outputs['Color'], node_hair.inputs['Color'])
    links.new(node_hair.outputs['BSDF'], node_output.inputs['Surface'])

    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def import_aging_sequence_to_blender(seq_dir, out_path, total_frames, args):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames
    
    # 1. 建立 360 度旋轉攝影機軌道 (Empty 物件)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    rotation_target = bpy.context.object
    rotation_target.name = "Rotation_Anchor"
    
    camera = scene.camera
    if not camera:
        bpy.ops.object.camera_add(location=(0, -2.5, 0), rotation=(math.radians(90), 0, 0))
        camera = bpy.context.object
        scene.camera = camera
        
    camera.parent = rotation_target
    
    # 設定攝影機旋轉 Keyframe (Z 軸一圈)
    rotation_target.rotation_euler[2] = 0
    rotation_target.keyframe_insert(data_path="rotation_euler", frame=1)
    rotation_target.rotation_euler[2] = math.radians(360)
    rotation_target.keyframe_insert(data_path="rotation_euler", frame=total_frames + 1)
    
    # 讓旋轉曲線變成均速線性 (相容 Blender 5.0+ 新 API)
    if rotation_target.animation_data and rotation_target.animation_data.action:
        bpy.ops.object.select_all(action='DESELECT')
        rotation_target.select_set(True)
        bpy.context.view_layer.objects.active = rotation_target
        
        action_obj = rotation_target.animation_data.action
        fcurves = action_obj.curves if hasattr(action_obj, "curves") else getattr(action_obj, "fcurves", [])
        
        for fcurve in fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR'

    # 2. 先把場景原本預設的靜態 'hair_01' 隱藏，避免重疊
    base_hair = bpy.data.objects.get("hair_01")
    if base_hair:
        base_hair.hide_render = True
        base_hair.hide_viewport = True

    print(f"--- 開始循環載入並設定時序 Keyframes (共 {total_frames} 影格) ---")
    
    # 3. 依次匯入每一格的 npz 檔案並用 Keyframe 控制可見度
    # 獲取場景中唯一的主頭髮物件
    main_hair = bpy.data.objects.get("hair_01")
    if not main_hair:
        print("錯誤：場景中找不到 'hair_01' 物件！")
        return

    print(f"--- 開始循環載入並設定時序 Keyframes (共 {total_frames} 影格) ---")

    for frame in range(1, total_frames + 1):
        # 💡 自製動態時間軸進度條邏輯 (維持不變)
        percent = (frame / total_frames) * 100
        bar_length = 30  
        filled_length = int(bar_length * frame // total_frames)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        sys.stdout.write(f"\rImporting Frame: |{bar}| {frame}/{total_frames} ({percent:.1f}%)")
        sys.stdout.flush()

        # 1. 切換 Blender 當前的時間軸影格
        bpy.context.scene.frame_set(frame)

        # 2. 讀取當前影格的 npz 路徑
        npz_file = os.path.join(seq_dir, f"aging_{frame:03d}.npz")
        
        # 3. 呼叫我們剛才改好的「單一物件更新與半徑寫入」函數
        success = update_hair_single_object(npz_file, args)
        if not success:
            continue
            
        # 4. 💡【最核心改動】強迫 Blender 的幾何拓撲與屬性在這一格插入關鍵影格 (Keyframe)
        # 這樣一來，Blender 就會乖乖記住這唯一物件在每一格的顏色與毛髮半徑，髮型完全不會走鐘！
        main_hair.data.keyframe_insert(data_path="geometry", frame=frame)

    print("\n--- 3D 時序資料匯入完成，準備啟動背景渲染 ---")

    # 4. 設定影片渲染輸出 (FFMPEG / MP4)
    # scene.render.image_settings.file_format = 'FFMPEG_VIDEO'
    # scene.render.ffmpeg.format = 'MPEG4'
    # scene.render.ffmpeg.codec = 'H264'
    # scene.render.resolution_x = 1080  
    # scene.render.resolution_y = 1080
    # scene.render.filepath = os.path.join(out_path, "aging_360_rotation.mp4")


    # # 4. 改為輸出標準 PNG 圖片序列 (相容 Blender 5.0)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' 
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    
    # 建立一個專門放渲染圖的資料夾
    render_frames_dir = os.path.join(out_path, "rendered_frames")
    os.makedirs(render_frames_dir, exist_ok=True)
    
    # 設定輸出路徑開頭，Blender 會自動在後面補上四碼影格編號 (例如: frame_0001.png)
    scene.render.filepath = os.path.join(render_frames_dir, "frame_")

    
    # 5. 渲染配置 (使用 Cycles 渲染毛髮細節)
    scene.render.engine = 'CYCLES'
    if torch.cuda.is_available():
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        scene.cycles.device = 'GPU'
    scene.cycles.samples = 32  
    
    print(f"正在背景渲染 360 度老化影片，輸出路徑: {scene.render.filepath}")
    bpy.ops.render.render(animation=True, write_still=True)
    print("360 度老化漸變影片渲染完成！")


def main():
    print("=== Blender Main Pipeline Triggered ===")

    # 1. 參數解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', type=str, default=None) 
    parser.add_argument('--input_sequence_dir', type=str, default=None) 
    parser.add_argument('--total_frames', type=int, default=120)        
    
    parser.add_argument('--out_path', required=True) 
    parser.add_argument('--export_alembic', action='store_true') 
    parser.add_argument('-ss', '--strands_subsample', type=float, default=1.0)  
    parser.add_argument('-vs', '--vertex_subsample', type=float, default=1.0)  
    parser.add_argument('-ar', '--alembic_resolution', type=int, default=7) 
    parser.add_argument('-sh', '--shrinkwrap', action='store_true') 
    
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    # 2. 載入基礎模板檔案 (.blend)
    path_in_blend = os.path.join(path_cur_script, "./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    bpy.ops.wm.open_mainfile(filepath=path_in_blend)

    # 3. 邏輯分支：時序動畫流程
    if args.input_sequence_dir is not None:
        print(f"偵測到時序資料夾輸入，開始執行老化動畫流程...")
        import_aging_sequence_to_blender(
            seq_dir=args.input_sequence_dir, 
            out_path=args.out_path, 
            total_frames=args.total_frames,
            args=args
        )
    
    # 4. 邏輯分支：傳統靜態渲染流程
    elif args.input_npz is not None:
        print(f"偵測到單一檔案輸入，執行傳統靜態渲染流程...")
        
        obj = create_hair_frame_mesh(args.input_npz, 1, args)
        
        bpy.context.scene.render.engine = 'CYCLES'
        if torch.cuda.is_available():
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'

        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'RENDERED'

        print('saving 靜態 .blend')
        out_scene_path = os.path.join(args.out_path, "blender_scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=out_scene_path) 

        if args.export_alembic:
            out_path_alembic = os.path.join(args.out_path, "hair.abc")
            print("exporting hair to", out_path_alembic)
            export_alembic(out_path_alembic, args.alembic_resolution)
            
    else:
        print("錯誤：請至少提供 --input_npz 或 --input_sequence_dir 其中一個參數！")

if __name__ == '__main__':
    main()