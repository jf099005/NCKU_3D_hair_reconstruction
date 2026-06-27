
#run with 
# ~/blender-4.1.1-linux-x64/blender  -t 8 --background --python ./inference/npz2blender.py -- --input_npz <NPZ_PATH> --out_path <OUTPUT_PATH> --export_alembic



# 把.npz中的顏色連接到blender中的vertex上



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



# def apply_hair_mask_coloring(curves_data, points, mask_path):
#     """
#     自動計算邊界並將 2D Mask 精確映射到頭髮象限
#     """
#     if not os.path.exists(mask_path):
#         print(f"找不到 Mask 檔案: {mask_path}")
#         return

#     # 1. 載入圖片
#     mask_img = Image.open(mask_path).convert('RGB')
    
#     # 【關鍵】Blender 的 Y 軸正向通常是「後」，圖片 Y 是「下」
#     # 為了讓圖片左上角對應到頭髮的左前方，我們需要翻轉圖片或調整映射邏輯
#     # 這裡建議先翻轉，讓邏輯直覺化
#     mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)
#     mask_img = mask_img.transpose(Image.ROTATE_180)
    
#     mask_w, mask_h = mask_img.size
#     mask_data = np.array(mask_img) / 255.0

#     # 2. 獲取頂點維度
#     nr_strands, nr_pts_per_strand, _ = points.shape
    
#     # 3. 取得所有髮根 (Point 0) 的 3D 座標
#     roots = points[:, 0, :] 
#     root_x = roots[:, 0]
#     root_y = roots[:, 1] # 在 Blender 俯視圖中，Y 通常代表前後

#     # 4. 【核心改進】自動計算頭髮的實際範圍
#     # 這樣不論模型多大，圖片都會剛好鋪滿
#     min_x, max_x = root_x.min(), root_x.max()
#     min_y, max_y = root_y.min(), root_y.max()
    
#     print(f"頭髮邊界偵測: X({min_x:.3f} to {max_x:.3f}), Y({min_y:.3f} to {max_y:.3f})")

#     # 5. 將 3D 座標轉為 0.0 ~ 1.0 的比例，再轉為像素索引
#     # 避免除以 0 的錯誤
#     range_x = (max_x - min_x) if max_x != min_x else 1.0
#     range_y = (max_y - min_y) if max_y != min_y else 1.0

#     # 計算索引
#     ix = ((root_x - min_x) / range_x * (mask_w - 1)).astype(int)
#     iy = ((root_y - min_y) / range_y * (mask_h - 1)).astype(int)

#     # 確保不超出邊界
#     ix = np.clip(ix, 0, mask_w - 1)
#     iy = np.clip(iy, 0, mask_h - 1)

#     # 6. 提取顏色
#     strand_colors = mask_data[iy, ix] 

#     # 7. 寫入 Blender 屬性
#     attr_name = "VertexColor"
#     if attr_name not in curves_data.attributes:
#         color_attr = curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
#     else:
#         color_attr = curves_data.attributes[attr_name]

#     full_colors = np.ones((nr_strands * nr_pts_per_strand, 4))
#     full_colors[:, :3] = np.repeat(strand_colors, nr_pts_per_strand, axis=0)

#     color_attr.data.foreach_set("color", full_colors.flatten())
#     print("已依照四個象限完成顏色分佈。")


# def setup_hair_material_nodes(obj_name, attr_name="VertexColor"):
#     """
#     自動為指定物件建立材質並串接 Attribute 節點到 Base Color
#     """
#     obj = bpy.data.objects.get(obj_name)
#     if not obj:
#         print(f"找不到物件: {obj_name}")
#         return

#     # 1. 取得或建立材質
#     mat_name = "Hair_Mask_Material"
#     mat = bpy.data.materials.get(mat_name)
#     if mat is None:
#         mat = bpy.data.materials.new(name=mat_name)
    
#     mat.use_nodes = True
#     nodes = mat.node_tree.nodes
#     links = mat.node_tree.links
#     nodes.clear() # 清除舊節點重新建立

#     # 2. 建立節點
#     # 建立 Attribute 節點
#     node_attr = nodes.new(type='ShaderNodeAttribute')
#     node_attr.attribute_name = attr_name
#     node_attr.location = (-300, 300)

#     # 建立 Principled BSDF 節點
#     node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
#     node_bsdf.location = (0, 300)
#     # 稍微調整預設參數，讓頭髮看起來不那麼亮
#     node_bsdf.inputs['Roughness'].default_value = 0.7 

#     # 建立 Material Output 節點
#     node_output = nodes.new(type='ShaderNodeOutputMaterial')
#     node_output.location = (300, 300)

#     # 3. 串接節點
#     links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
#     links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

#     # 4. 將材質分配給物件
#     if len(obj.data.materials) == 0:
#         obj.data.materials.append(mat)
#     else:
#         obj.data.materials[0] = mat

#     print(f"材質節點串接完成：{obj_name} 已連結屬性 '{attr_name}'")


# def apply_hair_mask_with_gradient(curves_data, points, mask_path):
#     """
#     根據 Mask 決定髮根顏色，並讓顏色從髮根過渡到髮梢
#     """
#     if not os.path.exists(mask_path):
#         print(f"找不到 Mask 檔案: {mask_path}")
#         return

#     # 1. 載入 Mask
#     mask_img = Image.open(mask_path).convert('RGB')
#     mask_img = mask_img.transpose(Image.ROTATE_180)
#     mask_w, mask_h = mask_img.size
#     mask_data = np.array(mask_img) / 255.0

#     # 2. 數據維度
#     nr_strands, nr_pts_per_strand, _ = points.shape
#     attr_name = "VertexColor"
    
#     # 3. 獲取頂點屬性
#     color_attr = curves_data.attributes.get(attr_name) or \
#                  curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

#     # 4. 取得髮根位置並映射顏色
#     roots = points[:, 0, :] 
#     scale = 1.0 
#     ix = np.clip(((roots[:, 0] + (scale/2)) / scale * (mask_w - 1)).astype(int), 0, mask_w - 1)
#     iy = np.clip(((roots[:, 1] + (scale/2)) / scale * (mask_h - 1)).astype(int), 0, mask_h - 1)
    
#     # 這是每根頭髮的「初始底色」 (來自 Mask)
#     root_colors_from_mask = mask_data[iy, ix] 

#     # 5. 定義髮梢的目標顏色 (例如你想讓所有髮尾都變淺金色或深色)
#     # 你也可以在這裡設為 [0, 0, 0] 讓髮尾變黑
#     tip_target_color = np.array([0.1, 0.1, 0.1]) # 範例：髮尾統一過渡到深色

#     # 6. 計算所有頂點的顏色過渡
#     # t 是從 0.0 (髮根) 到 1.0 (髮梢) 的比例
#     t = np.linspace(0, 1, nr_pts_per_strand).reshape(-1, 1) # (pts, 1)
    
#     # 初始化總顏色矩陣 (RGBA)
#     full_colors = np.ones((nr_strands * nr_pts_per_strand, 4))

#     # 遍歷髮絲進行插值 (使用矩陣運算提速)
#     # 計算公式：Color = RootColor * (1-t) + TipColor * t
#     for s_idx in range(nr_strands):
#         start = s_idx * nr_pts_per_strand
#         end = start + nr_pts_per_strand
        
#         # 取得該根頭髮的 Mask 底色
#         r_col = root_colors_from_mask[s_idx]
        
#         # 進行過渡計算
#         strand_gradient = r_col * (1 - t) + tip_target_color * t
#         full_colors[start:end, :3] = strand_gradient

#     # 7. 寫入數據
#     color_attr.data.foreach_set("color", full_colors.flatten())
#     print("髮根 Mask + 全長過渡上色完成！")


# def apply_dual_mask_hair_coloring(curves_data, points, root_mask_path, tip_mask_path):
#     """
#     根據兩個 Mask 圖片決定髮根與髮梢顏色，並在兩者間過渡
#     """
#     # 1. 載入並檢查兩個 Mask
#     if not os.path.exists(root_mask_path) or not os.path.exists(tip_mask_path):
#         print("錯誤：找不到 Mask 檔案。")
#         return

#     def load_mask(path):
#         img = Image.open(path).convert('RGB')
#         return np.array(img) / 255.0, img.size

#     root_mask_data, (rw, rh) = load_mask(root_mask_path)
#     tip_mask_data, (tw, th) = load_mask(tip_mask_path)

#     # 2. 獲取數據維度
#     nr_strands, nr_pts, _ = points.shape
#     attr_name = "VertexColor"
#     color_attr = curves_data.attributes.get(attr_name) or \
#                  curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

#     # 3. 獲取投影座標 (髮根與髮梢的 3D 座標)
#     # 我們假設使用 X, Y 軸投影
#     roots = points[:, 0, :]    # 每根的第一個點
#     tips = points[:, -1, :]    # 每根的最後一個點
    
#     scale = 1.0  # 需根據模型大小調整
    
#     # 計算髮根在 Root Mask 的像素索引
#     rix = np.clip(((roots[:, 0] + (scale/2)) / scale * (rw - 1)).astype(int), 0, rw - 1)
#     riy = np.clip(((roots[:, 1] + (scale/2)) / scale * (rh - 1)).astype(int), 0, rh - 1)
    
#     # 計算髮梢在 Tip Mask 的像素索引
#     tix = np.clip(((tips[:, 0] + (scale/2)) / scale * (tw - 1)).astype(int), 0, tw - 1)
#     tiy = np.clip(((tips[:, 1] + (scale/2)) / scale * (th - 1)).astype(int), 0, th - 1)

#     # 4. 提取顏色
#     root_colors = root_mask_data[riy, rix] # 每根髮絲的起點色
#     tip_colors = tip_mask_data[tiy, tix]   # 每根髮絲的終點色

#     # 5. 執行插值與上色
#     full_colors = np.ones((nr_strands * nr_pts, 4))
#     t = np.linspace(0, 1, nr_pts).reshape(-1, 1) # 從 0 到 1 的漸層係數

#     for s_idx in range(nr_strands):
#         start = s_idx * nr_pts
#         end = start + nr_pts
        
#         # 混合：Color = Root * (1-t) + Tip * t
#         strand_colors = root_colors[s_idx] * (1 - t) + tip_colors[s_idx] * t
#         full_colors[start:end, :3] = strand_colors

#     # 6. 寫入數據
#     color_attr.data.foreach_set("color", full_colors.flatten())
#     print(f"雙 Mask 投影完成！根部：{os.path.basename(root_mask_path)}, 梢部：{os.path.basename(tip_mask_path)}")


# def save_spherized_reference(reference_img_path, output_path):
#     """
#     將參考圖轉換為球面展開格式 (Equirectangular) 並儲存
#     """
#     ref_img = Image.open(reference_img_path).convert('RGB')
#     # ref_img = ref_img.transpose(Image.ROTATE_180)
#     width, height = ref_img.size
#     ref_data = np.array(ref_img)
    
#     # 建立新的畫布數據
#     # 我們建立一個網格 (U, V)，範圍均為 0~1
#     u_grid, v_grid = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
#     # 這裡模擬你腳本中的映射邏輯：
#     # 將網格座標映射回原始圖片的採樣點
#     # 實際上，如果你只是想看圖片如何被「包」在球上，
#     # 這種轉換會產生類似於全景圖的視覺效果
    
#     # 建立一個簡單的魚眼/球面變形效果
#     x = (u_grid - 0.5) * 2
#     y = (v_grid - 0.5) * 2
#     r = np.sqrt(x**2 + y**2)
#     r = np.clip(r, 0, 1)
    
#     # 進行球面扭曲計算 (魚眼投影模擬)
#     phi = np.arctan2(y, x)
#     theta = r * (np.pi / 2)
    
#     new_u = 0.5 + (np.sin(theta) * np.cos(phi) * 0.5)
#     new_v = 0.5 + (np.sin(theta) * np.sin(phi) * 0.5)
    
#     ix = (new_u * (width - 1)).astype(int)
#     iy = (new_v * (height - 1)).astype(int)
    
#     warped_data = ref_data[iy, ix]
    
#     # 儲存圖片
#     warped_img = Image.fromarray(warped_data.astype('uint8'))
#     warped_img.save(output_path)
#     print(f"Warped 參考圖已儲存至: {output_path}")


# def apply_spherical_projection_coloring(curves_data, points, reference_img_path):
#     """
#     將參考圖 Warp 成球體，並依據頂點 Normal 方向投影顏色
#     """
#     if not os.path.exists(reference_img_path):
#         print(f"找不到參考圖: {reference_img_path}")
#         return

#     # 1. 載入參考圖
#     ref_img = Image.open(reference_img_path).convert('RGB')
#     ref_img = ref_img.transpose(Image.ROTATE_180)
#     ref_w, ref_h = ref_img.size
#     ref_data = np.array(ref_img) / 255.0

#     # 2. 獲取數據維度
#     nr_strands, nr_pts, _ = points.shape
#     flat_points = points.reshape(-1, 3)
#     num_total_points = flat_points.shape[0]

#     # 3. 計算中心點 (假設頭部中心在座標原點或點群中心)
#     center = np.mean(flat_points, axis=0)
    
#     # 4. 計算每個點的方向向量 (即 Normal)
#     directions = flat_points - center
#     # 正規化向量，使其長度為 1 (變成球面上的座標)
#     norms = np.linalg.norm(directions, axis=1, keepdims=True)
#     # 防止除以零
#     norms[norms == 0] = 1
#     unit_vectors = directions / norms

#     # 5. 球面座標轉換 (3D Vector -> 2D UV)
#     # 使用 atan2 和 asin 來計算經緯度映射
#     u = 0.5 + (np.arctan2(unit_vectors[:, 0], unit_vectors[:, 1]) / (2 * np.pi))
#     v = 0.5 + (np.arcsin(unit_vectors[:, 2]) / np.pi)

#     # 6. 映射到像素索引
#     ix = (u * (ref_w - 1)).astype(int)
#     iy = (v * (ref_h - 1)).astype(int)
    
#     # 確保索引不越界
#     ix = np.clip(ix, 0, ref_w - 1)
#     iy = np.clip(iy, 0, ref_h - 1)

#     # 7. 提取顏色並寫入 Blender
#     # 提取出的顏色形狀為 (num_total_points, 3)
#     projected_colors = ref_data[iy, ix]
    
#     # 增加 Alpha 通道
#     full_colors = np.ones((num_total_points, 4))
#     full_colors[:, :3] = projected_colors

#     # 寫入屬性
#     attr_name = "VertexColor"
#     color_attr = curves_data.attributes.get(attr_name) or \
#                  curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
    
#     color_attr.data.foreach_set("color", full_colors.flatten())
#     print(f"球面投影上色完成！參考圖：{os.path.basename(reference_img_path)}")



# def apply_spherical_grid_sample_coloring(curves_data, strand_points_world, reference_img_tensor):
#     """
#     使用 PyTorch grid_sample 進行球面投影上色
#     strand_points_world: (N, pts, 3) 的 Tensor
#     reference_img_tensor: (1, 3, H, W) 的 Tensor (類似 scalp_texture)
#     """
#     device = strand_points_world.device
#     nr_strands, nr_pts, _ = strand_points_world.shape

#     # --- 新增旋轉邏輯 ---
#     # dims=[2, 3] 代表對 H (高度) 與 W (寬度) 同時翻轉，等同旋轉 180 度
#     reference_img_tensor = torch.flip(reference_img_tensor, dims=[2, 3])
    
#     # 1. 攤平所有點並計算相對於中心的法線方向
#     flat_points = strand_points_world.view(-1, 3)
#     center = torch.mean(flat_points, dim=0)
#     directions = flat_points - center
    
#     # 2. 正規化向量 (Normal)
#     normals = F.normalize(directions, p=2, dim=-1)
    
#     # 3. 轉換為球面座標 (Spherical Coordinates to UV)
#     # u = atan2(x, y) / 2pi + 0.5
#     # v = asin(z) / pi + 0.5
#     u = 0.5 + (torch.atan2(normals[:, 0], normals[:, 1]) / (2 * torch.pi))
#     v = 0.5 + (torch.asin(normals[:, 2]) / torch.pi)
    
#     # 4. 準備 grid_sample 的輸入 (要求範圍在 -1 到 1 之間)
#     # grid 的 shape 必須是 (N, H_out, W_out, 2)
#     uv = torch.stack([u, v], dim=-1) * 2 - 1.0
#     grid = uv.view(1, 1, -1, 2) # (1, 1, 總點數, 2)
    
#     # 5. 執行採樣 (對應你程式碼中的 scalp_texture 採樣邏輯)
#     # mode='bilinear' 可以讓縮放後的顏色更平滑
#     sampled_colors = F.grid_sample(reference_img_tensor, grid, 
#                                    mode='bilinear', padding_mode='border', align_corners=True)
    
#     # 6. 整理顏色數據 (RGBA)
#     sampled_colors = sampled_colors.view(3, -1).t() # (總點數, 3)
#     ones = torch.ones((sampled_colors.shape[0], 1), device=device)
#     full_colors_rgba = torch.cat([sampled_colors, ones], dim=-1) # (總點數, 4)
    
#     # 7. 寫入屬性 (轉換回 CPU 以便 Blender 寫入)
#     attr_name = "VertexColor"
#     color_attr = curves_data.attributes.get(attr_name) or \
#                  curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
                 
#     color_attr.data.foreach_set("color", full_colors_rgba.cpu().numpy().flatten())
#     print("PyTorch 球面採樣上色完成。")


# def apply_spherical_grid_sample_coloring_r(curves_data, strand_points_world, reference_img_tensor):
#     """
#     僅使用髮根 (Hair Roots) 進行球面投影採樣，並將該顏色應用於整根髮絲
#     strand_points_world: (N_strands, N_pts_per_strand, 3) 的 Tensor
#     reference_img_tensor: (1, 3, H, W) 的 Tensor
#     """
#     device = strand_points_world.device
#     nr_strands, nr_pts, _ = strand_points_world.shape

#     # 0. 旋轉圖片邏輯 (保持原有 180 度旋轉)
#     reference_img_tensor = torch.flip(reference_img_tensor, dims=[2, 3])
    
#     # 1. 【關鍵修改】只提取髮根點 (第 0 個點)
#     # roots shape: (nr_strands, 3)
#     roots = strand_points_world[:, 0, :]
    
#     # 計算中心點 (可以改為 torch.zeros_like(roots[0]) 如果要對齊世界原點)
#     center = torch.mean(roots, dim=0)
#     directions = roots - center
    
#     # 2. 正規化髮根向量 (Normal)
#     normals = F.normalize(directions, p=2, dim=-1)
    
#     # 3. 轉換髮根為球面座標 (UV)
#     u = 0.5 + (torch.atan2(normals[:, 0], normals[:, 1]) / (2 * torch.pi))
#     v = 0.5 + (torch.asin(normals[:, 2]) / torch.pi)
    
#     # 4. 準備 grid_sample 輸入
#     # uv shape: (nr_strands, 2), 範圍轉為 -1 到 1
#     uv = torch.stack([u, v], dim=-1) * 2 - 1.0
#     grid = uv.view(1, 1, nr_strands, 2) # (1, 1, 髮絲數量, 2)
    
#     # 5. 執行採樣 (每個髮絲採樣一個顏色)
#     # sampled_colors shape: (1, 3, 1, nr_strands)
#     sampled_colors = F.grid_sample(reference_img_tensor, grid, 
#                                    mode='bilinear', padding_mode='border', align_corners=True)
    
#     # 6. 整理並擴展顏色數據
#     # 將顏色從 (3, nr_strands) 轉為 (nr_strands, 3)
#     strand_colors_rgb = sampled_colors.view(3, nr_strands).t() 
    
#     # 【關鍵修改】使用 repeat_interleave 將髮根顏色擴展到整根髮絲的所有頂點
#     # 這樣一來，每根頭髮的 nr_pts 個點都會共用同一個髮根顏色
#     full_colors_rgb = torch.repeat_interleave(strand_colors_rgb, nr_pts, dim=0) # (總點數, 3)
    
#     # 加入 Alpha 通道
#     ones = torch.ones((full_colors_rgb.shape[0], 1), device=device)
#     full_colors_rgba = torch.cat([full_colors_rgb, ones], dim=-1) # (總點數, 4)
    
#     # 7. 寫入 Blender 屬性
#     attr_name = "VertexColor"
#     color_attr = curves_data.attributes.get(attr_name) or \
#                  curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
                 
#     color_attr.data.foreach_set("color", full_colors_rgba.cpu().numpy().flatten())
#     print(f"PyTorch 球面採樣完成：已根據 {nr_strands} 個髮根位置決定髮絲顏色。")


# def create_blended_reference_texture(image_paths, mask_paths, blur_radius=15, device="cuda"):
#     """
#     將多張參考圖根據遮罩拼貼，並進行邊緣模糊過渡。
#     image_paths: 參考圖路徑列表
#     mask_paths: 對應的遮罩圖路徑列表
#     blur_radius: 高斯模糊的半徑，數值越大邊緣越柔和
#     """
#     if len(image_paths) != len(mask_paths):
#         raise ValueError("圖片數量與遮罩數量不匹配")

#     final_texture = None
#     total_mask_weight = None
    
#     # 預期輸出尺寸 (以第一張圖為準)
#     first_img = Image.open(image_paths[0]).convert('RGB')
#     w, h = first_img.size
    
#     transform = transforms.Compose([
#         transforms.Resize((h, w)),
#         transforms.ToTensor()
#     ])

#     for img_p, mask_p in zip(image_paths, mask_paths):
#         # 1. 載入圖片與遮罩
#         img = Image.open(img_p).convert('RGB')
#         mask = Image.open(mask_p).convert('L') # 轉為灰度圖作為遮罩
        
#         # 2. 對遮罩應用高斯模糊以實現柔和邊緣
#         if blur_radius > 0:
#             mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
#         # 3. 轉為 Tensor
#         img_t = transform(img).to(device).unsqueeze(0) # [1, 3, H, W]
#         mask_t = transforms.ToTensor()(mask).to(device).unsqueeze(0) # [1, 1, H, W]
        
#         # 4. 累加顏色 (圖片 * 遮罩)
#         if final_texture is None:
#             final_texture = img_t * mask_t
#             total_mask_weight = mask_t
#         else:
#             final_texture += img_t * mask_t
#             total_mask_weight += mask_t

#     # 5. 正規化 (防止遮罩重疊區域亮度過高)
#     # 避免除以零
#     total_mask_weight = torch.clamp(total_mask_weight, min=1e-6)
#     final_texture = final_texture / total_mask_weight
    
#     return final_texture


def setup_hair_material_nodes(obj_name, attr_name="VertexColor"):
    obj = bpy.data.objects.get(obj_name)
    if not obj: return

    mat_name = "Hair_Final_Material"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # 建立節點
    node_attr = nodes.new(type='ShaderNodeAttribute')
    node_attr.attribute_name = attr_name
    
    # 使用 Principled Hair BSDF (Blender 專業頭髮著色器)
    # 如果是舊版 Blender，請改回 ShaderNodeBsdfPrincipled
    node_hair = nodes.new(type='ShaderNodeBsdfHairPrincipled')
    node_hair.location = (0, 300)

    # 【加入這行】將頭髮著色參數強制設定為「直接顏色 (Color)」模式
    # 這樣 Blender 才會完全採用你從 UV 貼圖擷取過來的 RGB 數值
    if hasattr(node_hair, 'parametrization'):
        node_hair.parametrization = 'COLOR'
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 300)

    # ====================================================================
    # 加入以下參數來強化 PBR 物理光影與非等向性 (Anisotropy)
    # ====================================================================
    
    # 1. 粗糙度控制 (Roughness)
    # 縱向粗糙度控制高光沿著髮絲方向的長度，徑向粗糙度控制高光橫向的寬度
    node_hair.inputs['Roughness'].default_value = 0.3        # 讓高光順著髮絲拉長
    node_hair.inputs['Radial Roughness'].default_value = 0.4 # 讓橫向反射集中一點，維持髮絲立體感
    
    # 2. 毛鱗片偏移 (Offset)
    # 模擬真實頭髮表面的毛鱗片傾斜角度 (預設通常是 2 度)。
    # 這是讓光影根據髮絲方向產生「天使光環」位移的絕對關鍵。
    # node_hair.inputs['Offset'].default_value = 2.0           
    
    # 3. 表層油脂光澤 (Coat) 
    # 模擬頭髮上的自然油脂或護髮產品，會多出一層非常銳利且清脆的白色反光
    # node_hair.inputs['Coat'].default_value = 0.15

    # 4. 隨機性 (Randomness)
    # 賦予每一根髮絲微小的粗糙度差異，打破 CG 算圖常見的「塑膠死板感」
    node_hair.inputs['Random Roughness'].default_value = 0.15 
    
    # ====================================================================

    # 連結：將 Attribute 的顏色連到 Hair BSDF 的 Color (或 Tint)
    links.new(node_attr.outputs['Color'], node_hair.inputs['Color'])
    links.new(node_hair.outputs['BSDF'], node_output.inputs['Surface'])

    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat



def export_painted_mask(obj_name, texture_name, save_path):
    """
    從指定物件的材質中找到繪製的紋理並儲存為圖片
    """
    obj = bpy.data.objects.get(obj_name)
    if not obj or not obj.data.materials:
        print("找不到物件或材質")
        return

    # 遍歷材質節點找到目標 Image Texture
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    mask_node = next((n for n in nodes if n.type == 'TEX_IMAGE' and texture_name in n.image.name), None)

    if mask_node and mask_node.image:
        # 強制儲存圖片到指定路徑
        mask_node.image.filepath_raw = save_path
        mask_node.image.save()
        print(f"Mask 已匯出至: {save_path}")
    else:
        print(f"找不到名為 {texture_name} 的紋理節點")


def setup_render_settings(width=1024, height=1024, samples=128):
    """
    統一設定渲染參數
    """
    scene = bpy.context.scene
    
    # 1. 設定解析度 (像素)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100  # 設定為 100% 確保實際輸出為設定像素
    
    # 2. 設定渲染引擎品質 (Samples)
    # 這裡假設使用 Cycles 渲染引擎
    scene.cycles.samples = samples


import math
import bpy

def setup_camera_at_view(view_name):
    # 確保場景中有頭部物件 (假設名稱為 "hair_01")
    target_obj = bpy.data.objects.get("hair_01")
    if not target_obj:
        print("未找到目標物件 hair_01")
        return

    # 1. 取得或建立攝影機
    cam = bpy.data.objects.get("Camera")
    if not cam:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
    
    bpy.context.scene.camera = cam

    # 2. 加入 Track To 約束 (關鍵：這會強制攝影機永遠看著目標)
    if not cam.constraints.get("TrackTo"):
        constraint = cam.constraints.new(type='TRACK_TO')
        constraint.target = target_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

    # 3. 設定相對於目標的角度與距離
    dist = 2.5
    angles = {'front': 0, 'right': 90, 'back': 180, 'left': 270}
    angle_rad = math.radians(angles[view_name])
    
    # 計算位置：根據角度繞著目標轉
    x = dist * math.sin(angle_rad)
    y = -dist * math.cos(angle_rad)
    cam.location = (x, y, 1.0) # 1.0 是高度
    
    # 4. 強制更新場景 (重要！)
    bpy.context.view_layer.update()
    
    return cam

def render_all_views(out_path):
    views = ['front', 'right', 'back', 'left']
    for v in views:
        setup_camera_at_view(v)
        bpy.context.scene.render.filepath = os.path.join(out_path, f"hair_{v}.png")
        bpy.ops.render.render(write_still=True)


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
    strand_colors_raw = hair_geom.get("colors")


    subsample_nr_strands=False
    #removes randomly x amount of strands or X nr of vertices
    if args.strands_subsample!=1.0 or args.vertex_subsample!=1.0:
        subsample_nr_strands=True
    if subsample_nr_strands:
        # 1. 抽樣髮絲數量
        print("before ramoving random curves, points is ", points.shape) #nr_strands x nr_verts x3
        num_strands_to_keep = int(points.shape[0] * args.strands_subsample)
        strands_to_keep = np.random.choice(points.shape[0], num_strands_to_keep, replace=False)
        points = points[strands_to_keep, :, :].copy()
        print("after removing random curves, points is ", points.shape)

        # 2. 抽樣每根髮絲的頂點數量
        #removing verts now 
        nr_verts_to_skip=int(np.floor(1.0/args.vertex_subsample))
        print("nr_verts_to_skip",nr_verts_to_skip)
        points = points[:, ::nr_verts_to_skip, :].copy()
        print("after removing consecurive vertices, points is ", points.shape)

        # 3. 同步抽樣顏色
        if strand_colors_raw is not None:
            strand_colors_raw = strand_colors_raw[strands_to_keep]
            print(f"同步抽樣顏色完成，剩餘顏色數: {len(strand_colors_raw)}")
    
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
    print(f"最終幾何狀態: 髮絲數={nr_strands}, 每根點數={nr_points_per_strand}")

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

    
    ###################################################################################
    # 1. 讀取 .npz 裡的顏色數據
    # 確保你的 npz 裡有存入 'colors' (形狀為 nr_strands x 3)
    if strand_colors_raw is not None:
        if len(strand_colors_raw) != nr_strands:
            print(f"嚴重錯誤：顏色數量({len(strand_colors_raw)}) 與髮絲數量({nr_strands}) 不一致！")
            # 強制切片以匹配數量，防止報錯導致 Blender 崩潰
            strand_colors_raw = strand_colors_raw[:nr_strands]
        print(f"正在應用顏色屬性: 總頂點數={nr_strands * nr_points_per_strand}")
        
        # 2. 建立或取得屬性
        attr_name = "VertexColor"
        color_attr = curves_data.attributes.get(attr_name) or \
                     curves_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')

        # # 3. 將顏色分配給所有頂點
        # # 因為 colors 是每根頭髮一個顏色，我們要擴展到該根頭髮的所有點 (nr_points_per_strand)
        # # 建立 RGBA 陣列 (N_points * 4)
        # num_total_points = nr_strands * nr_points_per_strand
        # full_colors_rgba = np.ones((num_total_points, 4))
        
        # # 使用 numpy 的 repeat 快速擴展顏色
        # # 假設 strand_colors_raw 形狀是 (nr_strands, 3)
        # repeated_rgb = np.repeat(strand_colors_raw, nr_points_per_strand, axis=0)
        # full_colors_rgba[:, :3] = repeated_rgb

        # # 4. 寫入 Blender
        # color_attr.data.foreach_set("color", full_colors_rgba.flatten())

        # 3. 將顏色分配給所有頂點
        # 因為 colors 是每根頭髮一個顏色，我們要擴展到該根頭髮的所有點 (nr_points_per_strand)
        # 建立 RGBA 陣列 (N_points * 4)
        num_total_points = nr_strands * nr_points_per_strand
        full_colors_rgba = np.ones((num_total_points, 4))
        
        # 使用 numpy 的 repeat 快速擴展顏色
        repeated_rgb = np.repeat(strand_colors_raw, nr_points_per_strand, axis=0)
        
        # 【加入這行】將 sRGB 轉換為 Linear 色彩空間 (Gamma 2.2 解碼)
        # 防止 Blender 渲染時發生重複提亮的 Washed out 現象
        repeated_rgb = np.power(repeated_rgb, 2.2) 
        
        full_colors_rgba[:, :3] = repeated_rgb

        # 4. 寫入 Blender
        color_attr.data.foreach_set("color", full_colors_rgba.flatten())
        print("成功將 Scalp Mapping 顏色應用至髮絲頂點。")
    else:
        print("警告：.npz 中沒有 'colors' 數據，請檢查前段 img2hair 腳本。")

        # 連結材質球
        setup_hair_material_nodes("hair_01", attr_name="VertexColor")
    ###################################################################################


    if not args.shrinkwrap:
        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")



    # Update the viewport to reflect changes
    obj.data.update_tag()
    obj.modifiers.update()
    # bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.context.view_layer.update()


    ###################################################################################
    # 在 main() 函式最後，save_as_mainfile 之前加入：

    # 設定渲染引擎為 Cycles (頭髮表現較佳)
    bpy.context.scene.render.engine = 'CYCLES'
    
    # 如果有 GPU，自動開啟 GPU 渲染
    if torch.cuda.is_available():
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'

    # 設定視圖模式為「材質預覽」或「渲染模式」
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED' # 直接開啟渲染預覽

    # 呼叫剛才定義的材質設定函式，確保 VertexColor 與材質球連結
    setup_hair_material_nodes("hair_01", attr_name="VertexColor")
    ###################################################################################

    # render_all_views(args.out_path)

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

