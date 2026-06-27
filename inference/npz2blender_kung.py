
#run with 
# ~/blender-4.1.1-linux-x64/blender  -t 8 --background --python ./inference/npz2blender.py -- --input_npz <NPZ_PATH> --out_path <OUTPUT_PATH> --export_alembic



# 可以跑把提取多個髮色，並拼接成一個髮色參考圖，投影到3D model的vertex上



import bpy
from bpy.app.handlers import persistent
import bpy_extras
import os
import numpy as np
# from gloss import *
# import trimesh
# import json
import argparse
import sys
# import imageio.v3 as iio
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


path_cur_script=os.path.dirname(os.path.abspath(__file__))

def export_alembic(out_alembic_path, resolution):
    print("-------------------------------------------")
    
    # bpy.ops.outliner.item_activate(deselect_all=True)
    bpy.data.objects["hair_01"].select_set(True)
    # hair = bpy.context.active_object
    hair = bpy.data.objects["hair_01"]
    bpy.context.view_layer.objects.active=hair


    start=time.time()
    for modif in hair.modifiers:
        print("applying",modif.name)
        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.modifier_apply(modifier=modif.name)
    print("finished applying all geometry nodes")
    end=time.time()
    print("applying geometry nodes took", end-start)
    #shrinkwrap on the scalp (Wrong because it makes weird strands for the long hair)
    #default hair with t=8: 20s
    #default hair with t=16: 15s
    #50% strans with t=16: 6s

    #with shrinkwrap on the whole mesh
    #default hair with t=8: 43s
    #default hair with t=16: 34s
    #50% strans with t=16: 14s
    #50% strans, 50%points with t=16: 7s
    #50% strans, 25%points with t=16: 5s




    #conver particle
    bpy.ops.curves.convert_to_particle_system()

    # bpy.ops.outliner.item_activate(deselect_all=True)
    # bpy.context.space_data.context = 'PARTICLES'
    bpy.context.object.show_instancer_for_render = False
    bpy.context.object.show_instancer_for_viewport = False
    #I have no idea which one actually works to increase resolution so I change all
    # bpy.data.particles["ParticleSettings"].display_step = 7
    # bpy.data.particles["ParticleSettings"].hair_step = 7
    # bpy.data.particles["ParticleSettings"].render_step = 7
    bpy.data.particles["ParticleSettings"].display_step = resolution
    bpy.data.particles["ParticleSettings"].hair_step = resolution
    bpy.data.particles["ParticleSettings"].render_step = resolution
    

    #hide everything except scalp
    for obj in bpy.data.objects:
        print("obj", obj)
        if obj.name!="smplx_scalp_blender":
            obj.hide_render=True
            obj.hide_viewport=True
        else:
            print("smplx scalp blender doesn't get hidden")
    # for obj in bpy.scene.objects:
        # print("obj in scene", obj)



    bpy.data.objects['smplx_scalp_blender'].hide_render=False
    bpy.data.objects['smplx_scalp_blender'].hide_viewport=False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_render = False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_viewport = False
    bpy.data.objects["smplx_scalp_blender"].select_set(True)



    bpy.ops.wm.alembic_export(filepath=out_alembic_path, check_existing=False, start=1, end=1,selected=True, visible_objects_only=True, uvs=False, packuv=False, normals=False, use_instancing=False, global_scale=1.0, export_hair=True, export_particles=False, as_background_job=False, evaluation_mode='VIEWPORT', init_scene_frame_range=True)



def apply_hair_mask_coloring(curves_data, points, mask_path):
    """
    自動計算邊界並將 2D Mask 精確映射到頭髮象限
    """
    if not os.path.exists(mask_path):
        print(f"找不到 Mask 檔案: {mask_path}")
        return

    # 1. 載入圖片
    mask_img = Image.open(mask_path).convert('RGB')
    
    # 【關鍵】Blender 的 Y 軸正向通常是「後」，圖片 Y 是「下」
    # 為了讓圖片左上角對應到頭髮的左前方，我們需要翻轉圖片或調整映射邏輯
    # 這裡建議先翻轉，讓邏輯直覺化
    mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)
    mask_img = mask_img.transpose(Image.ROTATE_180)
    
    mask_w, mask_h = mask_img.size
    mask_data = np.array(mask_img) / 255.0

    # 2. 獲取頂點維度
    nr_strands, nr_pts_per_strand, _ = points.shape
    
    # 3. 取得所有髮根 (Point 0) 的 3D 座標
    roots = points[:, 0, :] 
    root_x = roots[:, 0]
    root_y = roots[:, 1] # 在 Blender 俯視圖中，Y 通常代表前後

    # 4. 【核心改進】自動計算頭髮的實際範圍
    # 這樣不論模型多大，圖片都會剛好鋪滿
    min_x, max_x = root_x.min(), root_x.max()
    min_y, max_y = root_y.min(), root_y.max()
    
    print(f"頭髮邊界偵測: X({min_x:.3f} to {max_x:.3f}), Y({min_y:.3f} to {max_y:.3f})")

    # 5. 將 3D 座標轉為 0.0 ~ 1.0 的比例，再轉為像素索引
    # 避免除以 0 的錯誤
    range_x = (max_x - min_x) if max_x != min_x else 1.0
    range_y = (max_y - min_y) if max_y != min_y else 1.0

    # 計算索引
    ix = ((root_x - min_x) / range_x * (mask_w - 1)).astype(int)
    iy = ((root_y - min_y) / range_y * (mask_h - 1)).astype(int)

    # 確保不超出邊界
    ix = np.clip(ix, 0, mask_w - 1)
    iy = np.clip(iy, 0, mask_h - 1)

    # 6. 提取顏色
    strand_colors = mask_data[iy, ix] 

    # 7. 寫入 Blender 屬性
    attr_name = "VertexColor"
    if attr_name not in curves_data.attributes:
        color_attr = curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
    else:
        color_attr = curves_data.attributes[attr_name]

    full_colors = np.ones((nr_strands * nr_pts_per_strand, 4))
    full_colors[:, :3] = np.repeat(strand_colors, nr_pts_per_strand, axis=0)

    color_attr.data.foreach_set("color", full_colors.flatten())
    print("已依照四個象限完成顏色分佈。")


def setup_hair_material_nodes(obj_name, attr_name="VertexColor"):
    """
    自動為指定物件建立材質並串接 Attribute 節點到 Base Color
    """
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"找不到物件: {obj_name}")
        return

    # 1. 取得或建立材質
    mat_name = "Hair_Mask_Material"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear() # 清除舊節點重新建立

    # 2. 建立節點
    # 建立 Attribute 節點
    node_attr = nodes.new(type='ShaderNodeAttribute')
    node_attr.attribute_name = attr_name
    node_attr.location = (-300, 300)

    # 建立 Principled BSDF 節點
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 300)
    # 稍微調整預設參數，讓頭髮看起來不那麼亮
    node_bsdf.inputs['Roughness'].default_value = 0.7 

    # 建立 Material Output 節點
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 300)

    # 3. 串接節點
    links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    # 4. 將材質分配給物件
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    print(f"材質節點串接完成：{obj_name} 已連結屬性 '{attr_name}'")


def apply_hair_mask_with_gradient(curves_data, points, mask_path):
    """
    根據 Mask 決定髮根顏色，並讓顏色從髮根過渡到髮梢
    """
    if not os.path.exists(mask_path):
        print(f"找不到 Mask 檔案: {mask_path}")
        return

    # 1. 載入 Mask
    mask_img = Image.open(mask_path).convert('RGB')
    mask_img = mask_img.transpose(Image.ROTATE_180)
    mask_w, mask_h = mask_img.size
    mask_data = np.array(mask_img) / 255.0

    # 2. 數據維度
    nr_strands, nr_pts_per_strand, _ = points.shape
    attr_name = "VertexColor"
    
    # 3. 獲取頂點屬性
    color_attr = curves_data.attributes.get(attr_name) or \
                 curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

    # 4. 取得髮根位置並映射顏色
    roots = points[:, 0, :] 
    scale = 1.0 
    ix = np.clip(((roots[:, 0] + (scale/2)) / scale * (mask_w - 1)).astype(int), 0, mask_w - 1)
    iy = np.clip(((roots[:, 1] + (scale/2)) / scale * (mask_h - 1)).astype(int), 0, mask_h - 1)
    
    # 這是每根頭髮的「初始底色」 (來自 Mask)
    root_colors_from_mask = mask_data[iy, ix] 

    # 5. 定義髮梢的目標顏色 (例如你想讓所有髮尾都變淺金色或深色)
    # 你也可以在這裡設為 [0, 0, 0] 讓髮尾變黑
    tip_target_color = np.array([0.1, 0.1, 0.1]) # 範例：髮尾統一過渡到深色

    # 6. 計算所有頂點的顏色過渡
    # t 是從 0.0 (髮根) 到 1.0 (髮梢) 的比例
    t = np.linspace(0, 1, nr_pts_per_strand).reshape(-1, 1) # (pts, 1)
    
    # 初始化總顏色矩陣 (RGBA)
    full_colors = np.ones((nr_strands * nr_pts_per_strand, 4))

    # 遍歷髮絲進行插值 (使用矩陣運算提速)
    # 計算公式：Color = RootColor * (1-t) + TipColor * t
    for s_idx in range(nr_strands):
        start = s_idx * nr_pts_per_strand
        end = start + nr_pts_per_strand
        
        # 取得該根頭髮的 Mask 底色
        r_col = root_colors_from_mask[s_idx]
        
        # 進行過渡計算
        strand_gradient = r_col * (1 - t) + tip_target_color * t
        full_colors[start:end, :3] = strand_gradient

    # 7. 寫入數據
    color_attr.data.foreach_set("color", full_colors.flatten())
    print("髮根 Mask + 全長過渡上色完成！")


def apply_dual_mask_hair_coloring(curves_data, points, root_mask_path, tip_mask_path):
    """
    根據兩個 Mask 圖片決定髮根與髮梢顏色，並在兩者間過渡
    """
    # 1. 載入並檢查兩個 Mask
    if not os.path.exists(root_mask_path) or not os.path.exists(tip_mask_path):
        print("錯誤：找不到 Mask 檔案。")
        return

    def load_mask(path):
        img = Image.open(path).convert('RGB')
        return np.array(img) / 255.0, img.size

    root_mask_data, (rw, rh) = load_mask(root_mask_path)
    tip_mask_data, (tw, th) = load_mask(tip_mask_path)

    # 2. 獲取數據維度
    nr_strands, nr_pts, _ = points.shape
    attr_name = "VertexColor"
    color_attr = curves_data.attributes.get(attr_name) or \
                 curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

    # 3. 獲取投影座標 (髮根與髮梢的 3D 座標)
    # 我們假設使用 X, Y 軸投影
    roots = points[:, 0, :]    # 每根的第一個點
    tips = points[:, -1, :]    # 每根的最後一個點
    
    scale = 1.0  # 需根據模型大小調整
    
    # 計算髮根在 Root Mask 的像素索引
    rix = np.clip(((roots[:, 0] + (scale/2)) / scale * (rw - 1)).astype(int), 0, rw - 1)
    riy = np.clip(((roots[:, 1] + (scale/2)) / scale * (rh - 1)).astype(int), 0, rh - 1)
    
    # 計算髮梢在 Tip Mask 的像素索引
    tix = np.clip(((tips[:, 0] + (scale/2)) / scale * (tw - 1)).astype(int), 0, tw - 1)
    tiy = np.clip(((tips[:, 1] + (scale/2)) / scale * (th - 1)).astype(int), 0, th - 1)

    # 4. 提取顏色
    root_colors = root_mask_data[riy, rix] # 每根髮絲的起點色
    tip_colors = tip_mask_data[tiy, tix]   # 每根髮絲的終點色

    # 5. 執行插值與上色
    full_colors = np.ones((nr_strands * nr_pts, 4))
    t = np.linspace(0, 1, nr_pts).reshape(-1, 1) # 從 0 到 1 的漸層係數

    for s_idx in range(nr_strands):
        start = s_idx * nr_pts
        end = start + nr_pts
        
        # 混合：Color = Root * (1-t) + Tip * t
        strand_colors = root_colors[s_idx] * (1 - t) + tip_colors[s_idx] * t
        full_colors[start:end, :3] = strand_colors

    # 6. 寫入數據
    color_attr.data.foreach_set("color", full_colors.flatten())
    print(f"雙 Mask 投影完成！根部：{os.path.basename(root_mask_path)}, 梢部：{os.path.basename(tip_mask_path)}")


def save_spherized_reference(reference_img_path, output_path):
    """
    將參考圖轉換為球面展開格式 (Equirectangular) 並儲存
    """
    ref_img = Image.open(reference_img_path).convert('RGB')
    # ref_img = ref_img.transpose(Image.ROTATE_180)
    width, height = ref_img.size
    ref_data = np.array(ref_img)
    
    # 建立新的畫布數據
    # 我們建立一個網格 (U, V)，範圍均為 0~1
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # 這裡模擬你腳本中的映射邏輯：
    # 將網格座標映射回原始圖片的採樣點
    # 實際上，如果你只是想看圖片如何被「包」在球上，
    # 這種轉換會產生類似於全景圖的視覺效果
    
    # 建立一個簡單的魚眼/球面變形效果
    x = (u_grid - 0.5) * 2
    y = (v_grid - 0.5) * 2
    r = np.sqrt(x**2 + y**2)
    r = np.clip(r, 0, 1)
    
    # 進行球面扭曲計算 (魚眼投影模擬)
    phi = np.arctan2(y, x)
    theta = r * (np.pi / 2)
    
    new_u = 0.5 + (np.sin(theta) * np.cos(phi) * 0.5)
    new_v = 0.5 + (np.sin(theta) * np.sin(phi) * 0.5)
    
    ix = (new_u * (width - 1)).astype(int)
    iy = (new_v * (height - 1)).astype(int)
    
    warped_data = ref_data[iy, ix]
    
    # 儲存圖片
    warped_img = Image.fromarray(warped_data.astype('uint8'))
    warped_img.save(output_path)
    print(f"Warped 參考圖已儲存至: {output_path}")


def apply_spherical_projection_coloring(curves_data, points, reference_img_path):
    """
    將參考圖 Warp 成球體，並依據頂點 Normal 方向投影顏色
    """
    if not os.path.exists(reference_img_path):
        print(f"找不到參考圖: {reference_img_path}")
        return

    # 1. 載入參考圖
    ref_img = Image.open(reference_img_path).convert('RGB')
    ref_img = ref_img.transpose(Image.ROTATE_180)
    ref_w, ref_h = ref_img.size
    ref_data = np.array(ref_img) / 255.0

    # 2. 獲取數據維度
    nr_strands, nr_pts, _ = points.shape
    flat_points = points.reshape(-1, 3)
    num_total_points = flat_points.shape[0]

    # 3. 計算中心點 (假設頭部中心在座標原點或點群中心)
    center = np.mean(flat_points, axis=0)
    
    # 4. 計算每個點的方向向量 (即 Normal)
    directions = flat_points - center
    # 正規化向量，使其長度為 1 (變成球面上的座標)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    # 防止除以零
    norms[norms == 0] = 1
    unit_vectors = directions / norms

    # 5. 球面座標轉換 (3D Vector -> 2D UV)
    # 使用 atan2 和 asin 來計算經緯度映射
    u = 0.5 + (np.arctan2(unit_vectors[:, 0], unit_vectors[:, 1]) / (2 * np.pi))
    v = 0.5 + (np.arcsin(unit_vectors[:, 2]) / np.pi)

    # 6. 映射到像素索引
    ix = (u * (ref_w - 1)).astype(int)
    iy = (v * (ref_h - 1)).astype(int)
    
    # 確保索引不越界
    ix = np.clip(ix, 0, ref_w - 1)
    iy = np.clip(iy, 0, ref_h - 1)

    # 7. 提取顏色並寫入 Blender
    # 提取出的顏色形狀為 (num_total_points, 3)
    projected_colors = ref_data[iy, ix]
    
    # 增加 Alpha 通道
    full_colors = np.ones((num_total_points, 4))
    full_colors[:, :3] = projected_colors

    # 寫入屬性
    attr_name = "VertexColor"
    color_attr = curves_data.attributes.get(attr_name) or \
                 curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
    
    color_attr.data.foreach_set("color", full_colors.flatten())
    print(f"球面投影上色完成！參考圖：{os.path.basename(reference_img_path)}")



def apply_spherical_grid_sample_coloring(curves_data, strand_points_world, reference_img_tensor):
    """
    使用 PyTorch grid_sample 進行球面投影上色
    strand_points_world: (N, pts, 3) 的 Tensor
    reference_img_tensor: (1, 3, H, W) 的 Tensor (類似 scalp_texture)
    """
    device = strand_points_world.device
    nr_strands, nr_pts, _ = strand_points_world.shape

    # --- 新增旋轉邏輯 ---
    # dims=[2, 3] 代表對 H (高度) 與 W (寬度) 同時翻轉，等同旋轉 180 度
    reference_img_tensor = torch.flip(reference_img_tensor, dims=[2, 3])
    
    # 1. 攤平所有點並計算相對於中心的法線方向
    flat_points = strand_points_world.view(-1, 3)
    center = torch.mean(flat_points, dim=0)
    directions = flat_points - center
    
    # 2. 正規化向量 (Normal)
    normals = F.normalize(directions, p=2, dim=-1)
    
    # 3. 轉換為球面座標 (Spherical Coordinates to UV)
    # u = atan2(x, y) / 2pi + 0.5
    # v = asin(z) / pi + 0.5
    u = 0.5 + (torch.atan2(normals[:, 0], normals[:, 1]) / (2 * torch.pi))
    v = 0.5 + (torch.asin(normals[:, 2]) / torch.pi)
    
    # 4. 準備 grid_sample 的輸入 (要求範圍在 -1 到 1 之間)
    # grid 的 shape 必須是 (N, H_out, W_out, 2)
    uv = torch.stack([u, v], dim=-1) * 2 - 1.0
    grid = uv.view(1, 1, -1, 2) # (1, 1, 總點數, 2)
    
    # 5. 執行採樣 (對應你程式碼中的 scalp_texture 採樣邏輯)
    # mode='bilinear' 可以讓縮放後的顏色更平滑
    sampled_colors = F.grid_sample(reference_img_tensor, grid, 
                                   mode='bilinear', padding_mode='border', align_corners=True)
    
    # 6. 整理顏色數據 (RGBA)
    sampled_colors = sampled_colors.view(3, -1).t() # (總點數, 3)
    ones = torch.ones((sampled_colors.shape[0], 1), device=device)
    full_colors_rgba = torch.cat([sampled_colors, ones], dim=-1) # (總點數, 4)
    
    # 7. 寫入屬性 (轉換回 CPU 以便 Blender 寫入)
    attr_name = "VertexColor"
    color_attr = curves_data.attributes.get(attr_name) or \
                 curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
                 
    color_attr.data.foreach_set("color", full_colors_rgba.cpu().numpy().flatten())
    print("PyTorch 球面採樣上色完成。")


def apply_spherical_grid_sample_coloring_r(curves_data, strand_points_world, reference_img_tensor):
    """
    僅使用髮根 (Hair Roots) 進行球面投影採樣，並將該顏色應用於整根髮絲
    strand_points_world: (N_strands, N_pts_per_strand, 3) 的 Tensor
    reference_img_tensor: (1, 3, H, W) 的 Tensor
    """
    device = strand_points_world.device
    nr_strands, nr_pts, _ = strand_points_world.shape

    # 0. 旋轉圖片邏輯 (保持原有 180 度旋轉)
    reference_img_tensor = torch.flip(reference_img_tensor, dims=[2, 3])
    
    # 1. 【關鍵修改】只提取髮根點 (第 0 個點)
    # roots shape: (nr_strands, 3)
    roots = strand_points_world[:, 0, :]
    
    # 計算中心點 (可以改為 torch.zeros_like(roots[0]) 如果要對齊世界原點)
    center = torch.mean(roots, dim=0)
    directions = roots - center
    
    # 2. 正規化髮根向量 (Normal)
    normals = F.normalize(directions, p=2, dim=-1)
    
    # 3. 轉換髮根為球面座標 (UV)
    u = 0.5 + (torch.atan2(normals[:, 0], normals[:, 1]) / (2 * torch.pi))
    v = 0.5 + (torch.asin(normals[:, 2]) / torch.pi)
    
    # 4. 準備 grid_sample 輸入
    # uv shape: (nr_strands, 2), 範圍轉為 -1 到 1
    uv = torch.stack([u, v], dim=-1) * 2 - 1.0
    grid = uv.view(1, 1, nr_strands, 2) # (1, 1, 髮絲數量, 2)
    
    # 5. 執行採樣 (每個髮絲採樣一個顏色)
    # sampled_colors shape: (1, 3, 1, nr_strands)
    sampled_colors = F.grid_sample(reference_img_tensor, grid, 
                                   mode='bilinear', padding_mode='border', align_corners=True)
    
    # 6. 整理並擴展顏色數據
    # 將顏色從 (3, nr_strands) 轉為 (nr_strands, 3)
    strand_colors_rgb = sampled_colors.view(3, nr_strands).t() 
    
    # 【關鍵修改】使用 repeat_interleave 將髮根顏色擴展到整根髮絲的所有頂點
    # 這樣一來，每根頭髮的 nr_pts 個點都會共用同一個髮根顏色
    full_colors_rgb = torch.repeat_interleave(strand_colors_rgb, nr_pts, dim=0) # (總點數, 3)
    
    # 加入 Alpha 通道
    ones = torch.ones((full_colors_rgb.shape[0], 1), device=device)
    full_colors_rgba = torch.cat([full_colors_rgb, ones], dim=-1) # (總點數, 4)
    
    # 7. 寫入 Blender 屬性
    attr_name = "VertexColor"
    color_attr = curves_data.attributes.get(attr_name) or \
                 curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
                 
    color_attr.data.foreach_set("color", full_colors_rgba.cpu().numpy().flatten())
    print(f"PyTorch 球面採樣完成：已根據 {nr_strands} 個髮根位置決定髮絲顏色。")


def create_blended_reference_texture(image_paths, mask_paths, blur_radius=15, device="cuda"):
    """
    將多張參考圖根據遮罩拼貼，並進行邊緣模糊過渡。
    image_paths: 參考圖路徑列表
    mask_paths: 對應的遮罩圖路徑列表
    blur_radius: 高斯模糊的半徑，數值越大邊緣越柔和
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError("圖片數量與遮罩數量不匹配")

    final_texture = None
    total_mask_weight = None
    
    # 預期輸出尺寸 (以第一張圖為準)
    first_img = Image.open(image_paths[0]).convert('RGB')
    w, h = first_img.size
    
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])

    for img_p, mask_p in zip(image_paths, mask_paths):
        # 1. 載入圖片與遮罩
        img = Image.open(img_p).convert('RGB')
        mask = Image.open(mask_p).convert('L') # 轉為灰度圖作為遮罩
        
        # 2. 對遮罩應用高斯模糊以實現柔和邊緣
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 3. 轉為 Tensor
        img_t = transform(img).to(device).unsqueeze(0) # [1, 3, H, W]
        mask_t = transforms.ToTensor()(mask).to(device).unsqueeze(0) # [1, 1, H, W]
        
        # 4. 累加顏色 (圖片 * 遮罩)
        if final_texture is None:
            final_texture = img_t * mask_t
            total_mask_weight = mask_t
        else:
            final_texture += img_t * mask_t
            total_mask_weight += mask_t

    # 5. 正規化 (防止遮罩重疊區域亮度過高)
    # 避免除以零
    total_mask_weight = torch.clamp(total_mask_weight, min=1e-6)
    final_texture = final_texture / total_mask_weight
    
    return final_texture





def main():
    print("main")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True) #npz file to read and create a alembic from
    parser.add_argument('--out_path', required=True) #output path for the blender file and the alembic
    parser.add_argument('--export_alembic', action='store_true') #set it to true to also export an alembic file
    parser.add_argument('-ss', '--strands_subsample', type=float, default=1.0)  # perentage of strands we keep (1.0=keep all, 0.5=keep half, 0.25=keep quarter)
    parser.add_argument('-vs', '--vertex_subsample', type=float, default=1.0)  # perentage of vertices per strand to keep (1.0=keep all, 0.5=keep half, 0.25=keep quarter)
    parser.add_argument('-ar', '--alembic_resolution', type=int, default=7) #the resolution of the alembic, higher number means more points per strand (default=7 which is probably 2^7=128 points per strands)
    parser.add_argument('-sh', '--shrinkwrap', action='store_true') #set it to true to perform a shrinkwrap of the hair so that it avoids penetrating through the body
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    print("strands_subsample", args.strands_subsample)
    print("vertex_subsample", args.vertex_subsample)
    print("alembic_resolution", args.alembic_resolution)
    print("shrinkwrap", args.shrinkwrap)


    #read npz 
    path_hair=args.input_npz
    do_export_alembic=args.export_alembic


    hair_geom=np.load(path_hair)
    points=hair_geom["positions"] #nr_strands x nr_points_per_strand x 3


    subsample_nr_strands=False
    #removes randomly x amount of strands or X nr of vertices
    if args.strands_subsample!=1.0 or args.vertex_subsample!=1.0:
        subsample_nr_strands=True
    if subsample_nr_strands:
        print("before ramoving random curves, points is ", points.shape) #nr_strands x nr_verts x3
        num_strands_to_keep = int(points.shape[0] * args.strands_subsample)
        strands_to_keep = np.random.choice(points.shape[0], num_strands_to_keep, replace=False)
        points = points[strands_to_keep, :, :].copy()
        print("after removing random curves, points is ", points.shape)

        #removing verts now 
        nr_verts_to_skip=int(np.floor(1.0/args.vertex_subsample))
        print("nr_verts_to_skip",nr_verts_to_skip)
        points = points[:, ::nr_verts_to_skip, :].copy()
        print("after removing consecurive vertices, points is ", points.shape)
    print("final points", points.shape)

    #open the blender file
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v24.blend")
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v25_with_shrinkwrap.blend")
    # if args.shrinkwrap:
    #     path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    # else:
    #     path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v24.blend")
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v27_blender36.blend")
    path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    bpy.ops.wm.open_mainfile(filepath=path_in_blend)


    #write new geometry
    print("creating geometry")
    bpy.data.objects["hair_01"].select_set(True)
    obj = bpy.data.objects.get("hair_01")
    bpy.context.view_layer.objects.active = obj
    # bpy.ops.object.mode_set(mode='EDIT')
    # #  Get the evaluated state of the object to account for geometry nodes and modifiers
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # depsgraph.update()
    # eval_obj = obj.evaluated_get(depsgraph)
    # curves_data=eval_obj.data
    # help(obj.data)
    curves_data=obj.data
    nr_strands=points.shape[0]
    # nr_strands=3000
    nr_points_per_strand=points.shape[1]

    #v4 faster
    points_per_curve = [nr_points_per_strand for i in range(nr_strands)]
    curves_data.add_curves(points_per_curve)
    # print("added curves")
    # exit(1)

    # Prepare a flat array for positions
    flat_points = points.reshape(-1, 3)  # Flatten points to a 2D array
    flat_points[:, [1, 2]] = flat_points[:, [2, 1]]  # Swap y and z
    flat_points[:, 1] *= -1  # Negate the y values

    # Assign the flat array directly
    curves_data.points.foreach_set("position", flat_points.flatten())

    # --- 呼叫 Mask 上色 ---
    # 請確保你的 assets 資料夾裡有一張名為 hair_mask.png 的圖
    # mask_path = os.path.join(path_cur_script, "..", "samples", "hair_color", "hc1.png")
    # mask_path = os.path.join(path_cur_script, "..", "outputs_inference", "intermediates", "hair_color", "hair_color_FINAL_MIXED.png")
    # apply_hair_mask_coloring(curves_data, points, mask_path)
    # apply_hair_mask_with_gradient(curves_data, points, mask_path)

    # root_m = os.path.join(path_cur_script, "..", "samples", "hair_color", "hc7.png")
    root_m = os.path.join(path_cur_script, "..", "outputs_inference", "intermediates", "hair_color", "hair_color_FINAL_MIXED.png")
    # tip_m = os.path.join(path_cur_script, "..", "samples", "hair_color", "hc2.png")
    # apply_dual_mask_hair_coloring(curves_data, points, root_m, tip_m)

    # output_warp_path = os.path.join(args.out_path, "debug_warped_ref.png")
    # save_spherized_reference(root_m, output_warp_path)

    # apply_spherical_projection_coloring(curves_data, points, root_m)


    if os.path.exists(root_m):
        # 1. 儲存 Warp 調試圖 (視覺化)
        output_warp_path = os.path.join(args.out_path, "debug_warped_ref.png")
        save_spherized_reference(root_m, output_warp_path)

        # 2. 準備 PyTorch 數據
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_tensor = torch.from_numpy(points).to(device).float()
        
        # 載入參考圖並轉為 NCHW Tensor
        ref_img = Image.open(root_m).convert('RGB')
        ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0).to(device)

        # 3. 呼叫球面投影函數 (使用剛才幫你寫的函數)
        # 注意：這裡傳入 points_tensor，函數內會自動處理方向向量
    #     apply_spherical_grid_sample_coloring(curves_data, points_tensor, ref_tensor)
        apply_spherical_grid_sample_coloring_r(curves_data, points_tensor, ref_tensor)
    else:
        print(f"警告：找不到參考圖 {root_m}，跳過上色。")


    # # hair color parameter setting
    # images = [
    #     os.path.join(path_cur_script, "..", "samples", "hair_color", "hc1.png"),
    #     os.path.join(path_cur_script, "..", "samples", "hair_color", "hc2.png"),
    #     os.path.join(path_cur_script, "..", "samples", "hair_color", "hc1.png"),
    #     os.path.join(path_cur_script, "..", "samples", "hair_color", "hc2.png")
    # ]
    # masks = [
    #     os.path.join(path_cur_script, "..", "samples", "hair_color_mask", "mask4.png"),
    #     os.path.join(path_cur_script, "..", "samples", "hair_color_mask", "mask1.png"),
    #     os.path.join(path_cur_script, "..", "samples", "hair_color_mask", "mask2.png"),
    #     os.path.join(path_cur_script, "..", "samples", "hair_color_mask", "mask3.png")
    # ]
    # blur_radius=20


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # points_tensor = torch.from_numpy(points).to(device).float()

    # # 1. 生成拼貼後的參考圖
    # blended_ref_tensor = create_blended_reference_texture(images, masks, blur_radius, device=device)

    # # (選擇性) 儲存拼貼後的圖供檢查
    # debug_img = transforms.ToPILImage()(blended_ref_tensor.squeeze(0).cpu())
    # debug_img.save(os.path.join(args.out_path, "hc_mixed.png"))

    # # 2. 進行球面投影上色
    # # 這裡我們傳入已經拼貼好的 blended_ref_tensor
    # apply_spherical_grid_sample_coloring(curves_data, points_tensor, blended_ref_tensor)

    
    # --- 新增：自動設定材質節點 ---
    setup_hair_material_nodes("hair_01")
    # ---------------------------

    if not args.shrinkwrap:
        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")



    # Update the viewport to reflect changes
    obj.data.update_tag()
    obj.modifiers.update()
    # bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.context.view_layer.update()



    #save blend file
    print('saving .blend')
    out_scene_path=os.path.join(args.out_path, "blender_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_scene_path) 
    print('finished saving .blend')


    if do_export_alembic:
        out_path_alembic=os.path.join(args.out_path, "hair.abc")
        print("exporting hair to", out_path_alembic)
        export_alembic(out_path_alembic, args.alembic_resolution)

   
if __name__ == '__main__':
    main() 

