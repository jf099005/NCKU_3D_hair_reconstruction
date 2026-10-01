#!/usr/bin/env python3

# 將顏色的color map寫道.npz檔案中輸入到blender中

import torch
import cv2
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import torchvision
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import os
from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
import numpy as np
import time
import numpy as np
from models.strand_codec import StrandCodec
from utils.strand_util import sample_strands_from_scalp_with_density
from utils.diffusion_utils import sample_images_cfg

import torch
import torch._dynamo
import torchvision
import sys
import os
from utils.vis_util import img_2_pca
import torchvision.transforms as T
import k_diffusion as K
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from data_loader.dataloader import DEFAULT_BODY_DATA_DIR, DiffLocksDataset
from data_loader.mesh_utils import tbn_space_to_world
VisionRunningMode = mp.tasks.vision.RunningMode
#https://www.reddit.com/r/learnpython/comments/1cxe5ag/need_help_with_mediapipe/
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

import argparse
import torchvision
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import math

torch.autograd.set_grad_enabled(False)

ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.path.pardir)) + "/"
BLENDER_PATH = "/home/kyh/blender/blender"

class Mediapipe():
    def __init__(self, mode):
        super(Mediapipe, self).__init__()

        self.mode=mode

        # base_options = python.BaseOptions(
        #     model_asset_path=os.path.join(SCRIPT_DIR,'./assets/face_landmarker.task'),
        #     delegate=mp.tasks.BaseOptions.Delegate.GPU
        #     )
        # options = vision.FaceLandmarkerOptions(
        #                                 running_mode=mode,
        #                                 base_options=base_options,
        #                                 output_face_blendshapes=False,
        #                                 output_facial_transformation_matrixes=False,
        #                                 num_faces=10,
        #                                 min_face_detection_confidence=0.1,
        #                                 min_face_presence_confidence=0.1,
        #                                 )
        # self.detector = vision.FaceLandmarker.create_from_options(options)

    #=====================================================================================
    #=============================== Revision Region =====================================
    #=====================================================================================

    # --- 1. 原有的臉部偵測器配置 ---
        base_options_face = python.BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR,'./assets/face_landmarker.task'),
            delegate=mp.tasks.BaseOptions.Delegate.GPU
        )
        options_face = vision.FaceLandmarkerOptions(
            running_mode=mode,
            base_options=base_options_face,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options_face)

        # --- 2. 新增髮型分割器配置 (Hair Segmenter) ---
        # 請確保 assets 資料夾內有 self_selection.tflite 或 hair_segmenter.tflite
        base_options_seg = python.BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR,'./assets/selfie_multiclass.tflite'),
            delegate=mp.tasks.BaseOptions.Delegate.CPU  # <--- 改成 CPU 試試看
        )
        options_seg = vision.ImageSegmenterOptions(
            running_mode=mode,
            base_options=base_options_seg,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options_seg)


    def run_segmentation(self, rgb_image_numpy):
        # 建立 MediaPipe 影像格式
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_numpy)
        segmentation_result = self.segmenter.segment(image)
        # 取得類別遮罩
        category_mask = segmentation_result.category_mask.numpy_view()
        return category_mask
    

    #=====================================================================================
    #======================== Revision Region End ========================================
    #=====================================================================================


    def run(self, rgb_image_numpy):
    
        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_numpy)
        detection_result = self.detector.detect(image)

        # STEP 4: Detect face landmarks from the input image.
        if self.mode==VisionRunningMode.IMAGE:
            detection_result = self.detector.detect(image)
        else:
            raise Exception("Sorry, not implemented") 
       

        if len (detection_result.face_landmarks) == 0:
            print('No face detected')
            return None, None
        
        # face_landmarks = detection_result.face_landmarks[0]

        # Find the largest face by bounding box area
        largest_face_index = -1
        max_area = 0

        print("number of faces detected", len(detection_result.face_landmarks))
        for i, landmarks in enumerate(detection_result.face_landmarks):
            x_min = min([lm.x for lm in landmarks])
            x_max = max([lm.x for lm in landmarks])
            y_min = min([lm.y for lm in landmarks])
            y_max = max([lm.y for lm in landmarks])

            # Compute bounding box area
            area = (x_max - x_min) * (y_max - y_min)

            if area > max_area:
                max_area = area
                largest_face_index = i

        if largest_face_index == -1:
            print("No valid face detected")
            return None, None
        
        print("largest_face_index",largest_face_index)

        # Get the landmarks of the largest face
        face_landmarks = detection_result.face_landmarks[largest_face_index]


        face_landmarks_numpy = np.zeros((478, 3))

        for i, landmark in enumerate(face_landmarks):
            face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

        # face_landmarks = [(int(lm[0] * img_w), int(lm[1] * img_h)) for lm in face_landmarks]
        fm = []
        for i, landmark in enumerate(face_landmarks):
            fm.append((landmark.x, landmark.y))

        return face_landmarks_numpy, fm



def crop_face(image, face_landmarks, output_size, crop_size_multiplier=2.8):
    img_h, img_w, _ = image.shape


    if face_landmarks is None:
        return cv2.resize(image, (output_size, output_size))
    

    #v4==========
    # Convert normalized landmarks to pixel coordinates
    face_landmarks_px = [(int(lm[0] * img_w), int(lm[1] * img_h)) for lm in face_landmarks]

    # Key face points
    chin = face_landmarks_px[152]  # Chin point
    forehead = face_landmarks_px[10]  # Forehead point
    left_cheek = face_landmarks_px[234]  # Left cheek
    right_cheek = face_landmarks_px[454]  # Right cheek

    # Calculate face bounding box dimensions
    face_width = right_cheek[0] - left_cheek[0]
    face_height = chin[1] - forehead[1]

    # Calculate new face height while maintaining aspect ratio
    crop_size = max(face_width, face_height)  # Ensure square crop around the face

    #make the crop slightly bigger
    # crop_size=int(crop_size*2.8)
    crop_size=int(crop_size*crop_size_multiplier)

    # Calculate crop center
    face_center_x = (left_cheek[0] + right_cheek[0]) // 2
    # face_center_y = (forehead[1] + chin[1]) // 2
    face_center_y = int(forehead[1]*0.4 + chin[1]*0.6)  #not the middle of the face but more closer to the chin than the forehead


    # Crop boundaries in the original image
    crop_x1 = int(face_center_x - crop_size // 2)
    crop_x2 = crop_x1 + crop_size
    crop_y1 = int(face_center_y - crop_size // 2)
    crop_y2 = crop_y1 + crop_size


    #get how much in each direction do we need to pad with zeros
    pad_left=max(0, -crop_x1)
    pad_right=abs(min(0, img_w-crop_x2))
    pad_top=max(0, -crop_y1)
    pad_bottom=abs(min(0, img_h-crop_y2))


    # Extract the region from the original image
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

    cropped_region = image[crop_y1:crop_y2, crop_x1:crop_x2]


    #pad it with zeros so that the image is square
    padded_image = cv2.copyMakeBorder(
        cropped_region,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )

    # Decide interpolation method based on whether we’re upscaling or downscaling
    h, w = padded_image.shape[:2]
    if h > output_size or w > output_size:
        interpolation = cv2.INTER_AREA  # downscaling
    else:
        interpolation = cv2.INTER_CUBIC  # upscaling

    padded_image = cv2.resize(padded_image, (output_size, output_size), interpolation=interpolation)

    return padded_image



class DiffLocksInference():
    def __init__(self, path_ckpt_strandcodec, path_config_difflocks, path_ckpt_difflocks, path_ckpt_rgb2material=None, cfg_val=1.0, nr_iters_denoise=100, nr_chunks_decode=50):
        super(DiffLocksInference, self).__init__()

        self.nr_chunks_decode_strands=nr_chunks_decode
        self.nr_iters_denoise=nr_iters_denoise
        self.cfg_val=cfg_val


        self.mediapipe_img=Mediapipe(VisionRunningMode.IMAGE)

        #create dinov2 
        image_size=770 #nearest images size that divides cleanly by patch size 14
        self.dinov2_latents_preprocessor = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.dinov2_latents_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.dinov2_latents_model.cuda()

        #difflocks strand codec
        self.strand_codec = StrandCodec(do_vae=False, 
                        decode_type="dir",
                        scale_init=30.0,
                        nr_verts_per_strand=256, nr_values_to_decode=255,
                        dim_per_value_decoded=3).cuda()
        self.strand_codec.load_state_dict(torch.load(path_ckpt_strandcodec))
        # self.strand_codec.load_state_dict(torch.load(path_ckpt_strandcodec, map_location='cuda'), strict=False)
        self.strand_codec.eval()


        #difflocks diffusion
        config = K.config.load_config(path_config_difflocks)
        self.model_config = config['model']
        inner_model_ema = K.config.make_model(config).cuda()
        inner_model_ema.eval()
        model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)
        model_ema.eval()
        #IMPORTANT set the dropout rate here for the condition to whatever you need, to make it either conditional or unconditional
        # model_ema.inner_model.condition_dropout_rate=0.0
        # if state_path.exists() :
            # state = json.load(open(state_path))
            # ckpt_path = state['latest_checkpoint']
        print(f'Resuming from {path_ckpt_difflocks}...')
        ckpt = torch.load(path_ckpt_difflocks, map_location='cpu')
        model_ema.inner_model.load_state_dict(ckpt['model_ema'])
        del ckpt
        self.model_ema=model_ema.cuda()

        #rgb2material
        if path_ckpt_rgb2material is not None:
            self.rgb2material = RGB2MaterialModel(
                        input_dim=1024,
                        out_dim=11,
                        hidden_dim=64).cuda()
            self.rgb2material.load_state_dict(torch.load(path_ckpt_rgb2material))
            self.rgb2material.eval()
        else:
            self.rgb2material=None


        #hairsynth data
        self.normalization_dict=DiffLocksDataset.get_normalization_data()
        self.scalp_trimesh, self.scalp_mesh_data=DiffLocksDataset.compute_scalp_data(os.path.join(DEFAULT_BODY_DATA_DIR,"scalp.ply"))

    def rgb2hair(self, rgb_img, out_path=None):
        assert rgb_img.shape[1] == 3, "rgb_img needs to have 3 channels"
        assert len(rgb_img.shape) == 4, "rgb_img needs to be in format BCHW, so it needs 4 dimensions"

        if out_path is not None:
            os.makedirs(out_path,exist_ok=True)


        #run mediapipe on it
        #from BCHW to HW3
        frame=(rgb_img.permute(0,2,3,1).squeeze(0)*255.0).to(torch.uint8) 
        frame=frame.detach().cpu().numpy()
        print("frame",frame.shape)
        face_landmarks_px, face_landmarks = self.mediapipe_img.run(frame)
        if face_landmarks is None: 
            #there was no face detected, there is nothing to do
            return None
        frame=crop_face(frame, face_landmarks, output_size=770)

        #back to tensor
        img_tensor = torch.tensor(frame).cuda()
        img_tensor=img_tensor.permute(2,0,1).unsqueeze(0).float()/255.0
        rgb_img=img_tensor


        extra_args={}
        extra_args['latents_dict']={}

        print("rgb_img",rgb_img.shape)


        #dinov2 v2 
        rgb_input = self.dinov2_latents_preprocessor(rgb_img).to("cuda")
        dinov2_output = self.dinov2_latents_model.forward_features(rgb_input)
        patch_tok = dinov2_output["x_norm_patchtokens"].clone()
        cls_tok = dinov2_output["x_norm_clstoken"].clone()
        cls_token=cls_tok
        patch_embeddings = patch_tok
        #reshape to [Batch_size, h, w, embedding]
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        h = w = int(num_patches ** 0.5)  # Assuming the number of patches is a perfect square (e.g., 14x14)
        patch_embeddings_reshaped = patch_embeddings.reshape(batch_size, h, w, hidden_size)
        patch_embeddings_reshaped=patch_embeddings_reshaped.permute(0,3,1,2).contiguous() #Make it bchw 
        print("patch_embeddings_reshaped",patch_embeddings_reshaped.shape)

        extra_args['latents_dict']["dinov2"]={
                                    "cls_token": cls_token,
                                    "final_latent": patch_embeddings_reshaped,
                                    }
                                    

        #run diffusion
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): #we need this because flash attention only works with float16 and bfloat16
            scalp_texture = sample_images_cfg(1, 
                                              cfg_val=self.cfg_val, 
                                              cfg_interval=[0.0, 5.0], 
                                              model_ema=self.model_ema, 
                                              model_config=self.model_config, 
                                              nr_iters=self.nr_iters_denoise, 
                                              extra_args=extra_args)
        scalp_texture=scalp_texture.float()
        scalp_texture_orig=scalp_texture
        if out_path:
            np.savez(os.path.join(out_path, "scalp_texture.npz"), scalp_texture=scalp_texture_orig.cpu().numpy())
        print("scalp_texture", scalp_texture.shape)
        scalp_texture = scalp_texture_orig[:,0:-1,:,:] #get only the scalp texture part
        # scalp_texture_pca = img_2_pca(scalp_texture)
        # torchvision.utils.save_image(scalp_texture_pca.squeeze(0), "scalp_texture_sampled_diffusion.png")

        #density
        density_map=scalp_texture_orig[:,-1:,:,:]
        density_map=density_map*(0.5/self.model_config["sigma_data"]) + 0.5 
        density_map=density_map.clamp(0, 1)
        density_map[density_map<0.02]=0.0 #low density areas just set them to 0

        #increase density
        # density_map[density_map>0.02]+=1.0
        

        strand_points_world, strand_points_tbn = sample_strands_from_scalp_with_density(
            scalp_texture, 
            density_map, 
            self.strand_codec, 
            normalization_dict=self.normalization_dict, 
            scalp_mesh_data=self.scalp_mesh_data, 
            tbn_space_to_world_func=tbn_space_to_world, 
            nr_chunks=self.nr_chunks_decode_strands, upsample_multiplier=3)


        #get also material
        hair_material_dict=None
        if self.rgb2material is not None:
            rgb2mat_input_dict={}
            rgb2mat_input_dict["dinov2_latents"]=patch_embeddings_reshaped
            hair_material_dict = self.rgb2material(rgb2mat_input_dict)

            #save material
            if out_path:
                data={}
                if hair_material_dict is not None:
                    melanin=hair_material_dict["melanin"].item()
                    redness=hair_material_dict["redness"].item()
                    root_darkness_start=hair_material_dict["root_darkness_start"].item()
                    root_darkness_end=hair_material_dict["root_darkness_end"].item()
                    root_darkness_strength=hair_material_dict["root_darkness_strength"].item()
                    data["melanin"]=melanin
                    data["redness"]=redness
                    data["root_darkness_start"]=root_darkness_start
                    data["root_darkness_end"]=root_darkness_end
                    data["root_darkness_strength"]=root_darkness_strength

                path_json=os.path.join(out_path,"hair.json")
                with open(path_json, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
        
        if out_path:
            npz_out_path=os.path.join(out_path, "difflocks_output_strands.npz")
            np.savez(npz_out_path, positions=strand_points_world.cpu().numpy())

        #save also img
        if out_path:
            torchvision.utils.save_image(rgb_img, os.path.join(out_path, "rgb.png"))

        return strand_points_world, hair_material_dict



    def file2hair(self, file_path, out_path):
        frame=cv2.imread(file_path)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        rgb_img = torch.tensor(frame).cuda()
        rgb_img=rgb_img.permute(2,0,1).unsqueeze(0).float()/255.0


        strand_points_world, hair_material_dict = self.rgb2hair(rgb_img, out_path) 

        return strand_points_world, hair_material_dict
    

    #=====================================================================================
    #============================ Revision Region ========================================
    #=====================================================================================
    def _generate_texture_from_rgb(self, rgb_img):
        
        extra_args={}
        extra_args['latents_dict']={}
        
        # dinov2 v2 
        rgb_input = self.dinov2_latents_preprocessor(rgb_img).to("cuda")
        dinov2_output = self.dinov2_latents_model.forward_features(rgb_input)
        patch_tok = dinov2_output["x_norm_patchtokens"].clone()
        cls_tok = dinov2_output["x_norm_clstoken"].clone()
        cls_token=cls_tok
        patch_embeddings = patch_tok
        
        # Reshape patch tokens
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        h = w = int(num_patches ** 0.5)
        patch_embeddings_reshaped = patch_embeddings.reshape(batch_size, h, w, hidden_size)
        patch_embeddings_reshaped=patch_embeddings_reshaped.permute(0,3,1,2).contiguous() #Make it bchw 
        
        extra_args['latents_dict']["dinov2"]={
                                    "cls_token": cls_token, # <--- Global Cond
                                    "final_latent": patch_embeddings_reshaped,
                                    }
        
        # run diffusion
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scalp_texture_orig = sample_images_cfg(
                1, 
                cfg_val=self.cfg_val, 
                cfg_interval=[0.0, 5.0], 
                model_ema=self.model_ema, 
                model_config=self.model_config, 
                nr_iters=self.nr_iters_denoise, 
                extra_args=extra_args)
        scalp_texture_orig = scalp_texture_orig.float()
        
        return scalp_texture_orig, cls_token
    

    # 統一染色流程（舊版 median 作法）：用髮型分割抓出頭髮像素，取 RGB 中位數，
    # 生成一張 256x256 純色圖當作該張照片的 Color Map。
    def extract_color_map(self, frame_cropped):
        category_mask = self.mediapipe_img.run_segmentation(frame_cropped)
        hair_mask_2d = (category_mask == 1) # 類別 1 為頭髮
        hair_pixels = frame_cropped[hair_mask_2d]

        if len(hair_pixels) > 0:
            extracted_rgb = np.median(hair_pixels, axis=0).astype(np.uint8)
        else:
            extracted_rgb = np.array([50, 50, 50], dtype=np.uint8) # 預設深灰色

        color_img_256 = np.zeros((256, 256, 3), dtype=np.uint8)
        color_img_256[:] = extracted_rgb

        color_map_orig = torch.from_numpy(color_img_256).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return color_map_orig

    def files_to_scalp_textures(self, file_paths: list, out_path=None):
        all_texture_data = [] 
        
        for idx, file_path in enumerate(file_paths):
            print(f"Processing image {idx+1}/{len(file_paths)}: {file_path}")
            
            frame=cv2.imread(file_path)
            if frame is None:
                print(f"Warning: Could not read file {file_path}. Skipping.")
                continue

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            # mediapipe and crop_face
            # img_to_process = frame
            # rgb_img_tensor = torch.tensor(img_to_process).cuda()
            # rgb_img_tensor = rgb_img_tensor.permute(2,0,1).unsqueeze(0).float()/255.0
            
            # frame_mp=(rgb_img_tensor.permute(0,2,3,1).squeeze(0)*255.0).to(torch.uint8) 
            # frame_mp=frame_mp.detach().cpu().numpy()
            # face_landmarks_px, face_landmarks = self.mediapipe_img.run(frame_mp)
            # if face_landmarks is None: 
            #     print(f"No face detected in {file_path}. Skipping.")
            #     continue
            # frame_cropped=crop_face(frame_mp, face_landmarks, output_size=770)
            _, face_landmarks = self.mediapipe_img.run(frame)
            if face_landmarks is None: continue
            frame_cropped = crop_face(frame, face_landmarks, output_size=770)

            # Back to tensor for DINOv2
            rgb_img_cropped = torch.tensor(frame_cropped).cuda()
            rgb_img_cropped = rgb_img_cropped.permute(2,0,1).unsqueeze(0).float()/255.0

            # Generate Scalp Texture
            scalp_texture_orig, cls_token = self._generate_texture_from_rgb(rgb_img_cropped)

            # 統一染色流程（舊版 median 作法）
            color_map_orig = self.extract_color_map(frame_cropped)

            all_texture_data.append({
                "scalp_texture_orig": scalp_texture_orig,
                "cls_token": cls_token,
                "color_map_orig": color_map_orig,
                "input_file": file_path
            })

            if out_path:
                 os.makedirs(out_path, exist_ok=True)
                 np.savez(os.path.join(out_path, f"scalp_texture_{idx}.npz"), scalp_texture=scalp_texture_orig.cpu().numpy())
                 save_color_map(color_map_orig, out_path, f"color_map_orig_{idx+1}")

        return all_texture_data


    # def get_hair_root_colors(self, density_map_mixed, color_map_mixed):
    #     # 1. 取得密度遮罩
    #     mask = (density_map_mixed > 0.02).squeeze() 

    #     # 2. 核心修正：嘗試縱向掃描 (mask.T) 並配合 Y 軸翻轉 (255 - y)
    #     # 這是為了解決 2D 影像座標 (左上 0,0) 與 3D UV 座標 (左下 0,0) 的常見差異
    #     indices = torch.nonzero(mask.T)[:, [1, 0]] # 轉置掃描並換回 [y, x]
        
    #     target_y = 255 - indices[:, 0] # 嘗試 Y 軸翻轉採樣
    #     target_x = indices[:, 1]
        
    #     # 限制範圍防止越界
    #     target_y = torch.clamp(target_y, 0, 255)
    #     target_x = torch.clamp(target_x, 0, 255)

    #     # 3. 從 color_map 提取顏色 (B, C, H, W) -> (3, N)
    #     colors = color_map_mixed[0, :, target_y.long(), target_x.long()]
        
    #     return colors.permute(1, 0) # [N, 3]
        
    #=====================================================================================
    #======================== Revision Region End ========================================
    #=====================================================================================


def run():

    path_strand_codec=os.path.join(ROOT, "./checkpoints/strand_vae/strand_codec.pt")
    path_config = os.path.join(ROOT, "./configs/config_scalp_texture_conditional.json")
    path_diffusion_model_ckpt_path = os.path.join(ROOT, "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth") #longest trained one yet
    path_material_model_ckpt_path = os.path.join(ROOT, "./checkpoints/rgb2material/rgb2material.pt") 
    
    out_path="./outputs_inference/"

    difflocks= DiffLocksInference(
        path_strand_codec, 
        path_config, 
        path_diffusion_model_ckpt_path, 
        path_material_model_ckpt_path)


    #run----
    img_path="./samples/medium_11.png"
    strand_points_world, hair_material_dict=difflocks.file2hair(img_path, out_path) 
    print("hair_material_dict",hair_material_dict)


#=====================================================================================
#============================ Revision Region ========================================
#=====================================================================================

def visualize_scalp_texture(scalp_texture_latent: torch.Tensor, output_path: str, filename: str):
    """
    Assumption : The shape of scale texture = (1, 64, H, W)

    Perform dimensionality reduction on the 64-dimensional Scalp Texture latent code 
    and save the result as a PNG image.
    """
    
    if scalp_texture_latent.shape[1] == 65:
        T_latent = scalp_texture_latent[:, 0:64, :, :]
    elif scalp_texture_latent.shape[1] == 64:
        T_latent = scalp_texture_latent
    else:
        print(f"Warning: Latent size ({scalp_texture_latent.shape}) is not expected。")
        return

    # 1. Dimensionality reduction: Choose the first three channels
    # T_vis = img_2_pca(T_latent)
    T_vis = T_latent[:, 0:3, :, :] # choose tunnel 0, 1, 2 to be R, G, B
    
    # 2. Normalize to the range [0, 1]
    min_val = T_vis.amin(dim=(0, 2, 3), keepdim=True)
    max_val = T_vis.amax(dim=(0, 2, 3), keepdim=True)
    normalized_T = (T_vis - min_val) / (max_val - min_val + 1e-8)
    
    # 3. Store scalp texture picture
    full_path = os.path.join(output_path, f"{filename}.png")
    vutils.save_image(normalized_T, full_path)
    print(f"scalp texture picture store_path: {full_path}")


def save_grayscale_mask(mask_tensor: torch.Tensor, output_path: str, filename: str):
    """
    Store mask
    """
    
    # Ensure mask_tensor is in the (B, 1, H, W) format;
    # if not, attempt to adjust its shape/dimensions.
    if mask_tensor.dim() == 2: # only H, W
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 3: # (1, H, W) ot (H, W, 1)
        if mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor.permute(2, 0, 1).unsqueeze(0) # assume (H, W, 1) -> (1, 1, H, W)
        else:
            mask_tensor = mask_tensor.unsqueeze(0)
            
    # Clamp tensor values between 0 and 1
    mask_tensor = mask_tensor.clamp(0, 1)
    
    full_path = os.path.join(output_path, f"{filename}.png")
    # vutils.save_image requires the input tensor to be
    # in the (C, H, W) 或 (B, C, H, W)
    vutils.save_image(mask_tensor.cpu(), full_path)
    print(f"mask picture store_path: {full_path}")


def save_color_map(color_map_tensor: torch.Tensor, output_path: str, filename: str):
    """將混合後的 Color Map 張量儲存為 PNG 圖片檔案。"""
    if color_map_tensor is None:
        print("Warning: Cannot save color map, tensor is None.")
        return

    color_map_tensor = color_map_tensor.clamp(0.0, 1.0)

    full_path = os.path.join(output_path, f"{filename}.png")
    vutils.save_image(color_map_tensor.cpu(), full_path)
    print(f"Blended Color Map store_path: {full_path}")


def load_and_resize_mask(
        mask_path: str, 
        target_h: int, 
        target_w: int, 
        device: torch.device,
        blur_sigma: float = 3.0
        ) -> torch.Tensor:
    """
    load and resize mask
    """

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
    # 1. Read the image in grayscale mode
    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_np is None:
        raise ValueError(f"Unable to read the mask image: {mask_path}")
        
    # 2. Convert to PyTorch Tensor：HWC -> CHW (1, H, W)
    mask_tensor = torch.from_numpy(mask_np).float().to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    # 3. Normalization: Convert the 0-255 range to the 0.0-1.0 range
    mask_tensor /= 255.0
    
    # 4. Scaling: Scale to the dimensions of the Scalp Texture (e.g., 256x256)
    resized_mask = F.interpolate(
        mask_tensor, 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    )

    # 5. Gaussion blur
    if blur_sigma > 0:
        # Calculate the size of the Gaussian kernel
        # usually 4–6 times the sigma, and must be an odd number
        kernel_size = int(blur_sigma * 4 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        resized_mask = TF.gaussian_blur(resized_mask, 
                                         kernel_size=[kernel_size, kernel_size], 
                                         sigma=[blur_sigma, blur_sigma])
    
    # Ensure the mask is between 0 and 1
    return resized_mask.clamp(0, 1)


# --- 1. Generate multiple Scalp Texture ---
def scalp_texture(
        path_strand_codec,
        path_config,
        path_diffusion_model_ckpt_path,
        path_material_model_ckpt_path,
        input_file_paths,
        out_path,
        mask_file_paths
):
    print("--- 1. Generate multiple Scalp Texture ---")
    difflocks= DiffLocksInference(
        path_strand_codec, 
        path_config, 
        path_diffusion_model_ckpt_path, 
        path_material_model_ckpt_path
        )

    all_texture_data = difflocks.files_to_scalp_textures(
        file_paths=input_file_paths, 
        out_path=os.path.join(out_path, "intermediates")
    )
    
    T_list = [data['scalp_texture_orig'] for data in all_texture_data]
    C_list = [data['color_map_orig'] for data in all_texture_data]

    for i, T_orig in enumerate(T_list):
        visualize_scalp_texture(T_orig, out_path, f"scalp_texture_{i+1}")

    # print (f"T_list: {len(T_list)}")
    # print (f"mask: {len(mask_file_paths)}")
    if len(T_list) != len(mask_file_paths) + 1:
        print(f"Error: Input images ({len(T_list)}) "
              f"must be one more than masks ({len(mask_file_paths)}).")
        return

    return difflocks, T_list, C_list


# --- 2-1. Executing the Scalp Texture Blending Logic ---
def scalp_texture_blending(
        T_list,
        C_list,
        mask_file_paths,
        out_path,
        difflocks
):
    print("--- 2-1. Executing the Scalp Texture Blending Logic ---")

    # 1. Get size and device
    B, C, H, W = T_list[0].shape
    device = T_list[0].device
    # print(f"Scalp Texture shape: ({B}, {C}, {H}, {W})")

    # 2-2. Load mask
    M_list_T = []
    M_list_C = []
    BLUR_SIGMA_VALUE = 9.0
    try:
        for i, m_path in enumerate(mask_file_paths):
            M_list_T.append(load_and_resize_mask(m_path, H, W, device, blur_sigma=BLUR_SIGMA_VALUE))
            M_list_C.append(load_and_resize_mask(m_path, 256, 256, device, blur_sigma=BLUR_SIGMA_VALUE))
    except Exception as e:
        print(f"Mask load failed: {e}")
        return

    # 3. scalp texture blending logic
    # 3-1. Initialize the Base Layer (using T1 as the starting point)
    T_orig_mixed = T_list[0].clone()
    C_orig_mixed = C_list[0].clone()

    # 3-2. Iterative Loop : M1/T2, M2/T3, M3/T4 ...
    for i in range(len(M_list_T)):
        M_T = M_list_T[i]
        M_C = M_list_C[i]
        T_target = T_list[i+1]
        C_target = C_list[i+1]

        # save mask
        save_grayscale_mask(M_T, os.path.join(out_path, "masks"), f"mask_M{i+1}")

        M_expanded_T = M_T.expand_as(T_orig_mixed)
        M_expanded_C = M_C.expand_as(C_orig_mixed)

        # blending： T_orig_mixed = (1 - M) * T_current + M * T_target
        T_orig_mixed = (1.0 - M_expanded_T) * T_orig_mixed + M_expanded_T * T_target
        C_orig_mixed = (1.0 - M_expanded_C) * C_orig_mixed + M_expanded_C * C_target

        print(f"Used M{i+1} to blend T{i+2} into the result")

    # 3-3-1. Save blending result
    visualize_scalp_texture(T_orig_mixed, out_path, "scalp_texture_FINAL_MIXED")
    print("Finish Multiple Scalp Texture Blending")

    # 3-3-2. Save blending hair color result
    hair_color_dir = os.path.join(out_path, "intermediates", "hair_color")
    os.makedirs(hair_color_dir, exist_ok=True)
    save_color_map(C_orig_mixed, hair_color_dir, "hair_color_FINAL_MIXED")

    # 4. Blended Decoding Step
    # 4-1. Separate Scalp Texture (T) and Density Map (D)
    scalp_texture_mixed = T_orig_mixed[:, 0:-1, :, :] # latent code (T)
    density_map_mixed = T_orig_mixed[:, -1:, :, :]    # density map(D)

    # 4-2. Process the Density Map
    density_map_mixed = density_map_mixed * (0.5 / difflocks.model_config["sigma_data"]) + 0.5
    density_map_mixed = density_map_mixed.clamp(0, 1)
    density_map_mixed[density_map_mixed < 0.02] = 0.0


    return scalp_texture_mixed, density_map_mixed, C_orig_mixed


def scalp_texture_blending_traditional(
        T_list,
        C_list,
        mask_file_paths,
        out_path,
        difflocks
):
    print("--- 2-1. Executing the Scalp Texture Blending Logic (TRADITIONAL BASELINE) ---")

    # 1. Get size and device
    B, C, H, W = T_list[0].shape
    device = T_list[0].device

    # 2-2. Load mask
    M_list_T = []
    M_list_C = []
    BLUR_SIGMA_VALUE = 9.0

    # 計算 Gaussian Blur 的 Kernel Size
    k_size = int(BLUR_SIGMA_VALUE * 4 + 1)
    if k_size % 2 == 0: k_size += 1

    try:
        for i, m_path in enumerate(mask_file_paths):
            # 這裡即使讀取進來有模糊，我們稍後也會強制把它變成硬遮罩
            M_list_T.append(load_and_resize_mask(m_path, H, W, device, blur_sigma=0.0))
            M_list_C.append(load_and_resize_mask(m_path, 256, 256, device, blur_sigma=0.0))
    except Exception as e:
        print(f"Mask load failed: {e}")
        return

    # 3. scalp texture blending logic
    T_orig_mixed = T_list[0].clone()
    C_orig_mixed = C_list[0].clone()

    # 3-2. Iterative Loop
    for i in range(len(M_list_T)):
        M_T = M_list_T[i]
        M_C = M_list_C[i]
        T_target = T_list[i+1]
        C_target = C_list[i+1]
        
        M_expanded_T = M_T.expand_as(T_orig_mixed)
        M_expanded_C = M_C.expand_as(C_orig_mixed)
        
        # ---------------------------------------------------
        # ❌ 測試版 1：建立硬遮罩 (Hard Mask)
        # ---------------------------------------------------
        M_expanded_T_hard = (M_expanded_T > 0.01).float()
        M_expanded_C_hard = (M_expanded_C > 0.01).float()
        
        # ---------------------------------------------------
        # ❌ 測試版 2：硬拼接 (Hard Stitching)
        # ---------------------------------------------------
        T_hard_mixed = (1.0 - M_expanded_T_hard) * T_orig_mixed + M_expanded_T_hard * T_target
        C_hard_mixed = (1.0 - M_expanded_C_hard) * C_orig_mixed + M_expanded_C_hard * C_target
        
        # ---------------------------------------------------
        # ❌ 測試版 3：自動尋找「交界處」製作邊界遮罩 (Boundary Mask)
        # ---------------------------------------------------
        # 分別為 T (65通道) 和 C (3通道) 製作專屬的高斯模糊過渡帶
        M_soft_for_boundary_T = TF.gaussian_blur(M_expanded_T_hard, [k_size, k_size], [BLUR_SIGMA_VALUE, BLUR_SIGMA_VALUE])
        M_soft_for_boundary_C = TF.gaussian_blur(M_expanded_C_hard, [k_size, k_size], [BLUR_SIGMA_VALUE, BLUR_SIGMA_VALUE])
        
        # 分別算出對應通道數的邊界遮罩
        boundary_mask_T = 1.0 - torch.abs(M_soft_for_boundary_T * 2.0 - 1.0) 
        boundary_mask_C = 1.0 - torch.abs(M_soft_for_boundary_C * 2.0 - 1.0) 
        
        # ---------------------------------------------------
        # ❌ 測試版 4：局部特徵域模糊 (Localized Feature Blurring)
        # ---------------------------------------------------
        # 先算出一張「全糊」的圖
        T_blurred_full = TF.gaussian_blur(T_hard_mixed, [k_size, k_size], [BLUR_SIGMA_VALUE, BLUR_SIGMA_VALUE])
        C_blurred_full = TF.gaussian_blur(C_hard_mixed, [k_size, k_size], [BLUR_SIGMA_VALUE, BLUR_SIGMA_VALUE])
        
        # 最後：用各自專屬的邊界遮罩進行合成！
        T_orig_mixed = T_hard_mixed * (1.0 - boundary_mask_T) + T_blurred_full * boundary_mask_T
        C_orig_mixed = C_hard_mixed * (1.0 - boundary_mask_C) + C_blurred_full * boundary_mask_C
        
        print(f"Used M{i+1} to apply LOCALIZED blur on the seam of T{i+2} and C{i+2}")

    # 3-3-1. Save blending result
    visualize_scalp_texture(T_orig_mixed, out_path, "scalp_texture_FINAL_MIXED_TRADITIONAL")
    print("Finish Multiple Scalp Texture Blending (Traditional Baseline)")

    # 3-3-2. Save blending hair color result
    hair_color_dir = os.path.join(out_path, "intermediates", "hair_color")
    os.makedirs(hair_color_dir, exist_ok=True)
    save_color_map(C_orig_mixed, hair_color_dir, "hair_color_FINAL_MIXED_TRADITIONAL")

    # 4. Blended Decoding Step
    scalp_texture_mixed = T_orig_mixed[:, 0:-1, :, :] 
    density_map_mixed = T_orig_mixed[:, -1:, :, :]    
    
    density_map_mixed = density_map_mixed * (0.5 / difflocks.model_config["sigma_data"]) + 0.5 
    density_map_mixed = density_map_mixed.clamp(0, 1)
    density_map_mixed[density_map_mixed < 0.02] = 0.0 

    return scalp_texture_mixed, density_map_mixed, C_orig_mixed


# --- 2-2. Executing the Scalp Texture interpolating Logic ---
def scalp_texture_interpolating(
        T_list,
        C_list,
        mask_file_paths,
        out_path,
        difflocks
):  
    print("--- 2-2. Executing the Scalp Texture interpolating Logic ---")

    # 1-1. Get size and device
    B, C, H, W = T_list[0].shape
    device = T_list[0].device
    # print(f"Scalp Texture shape: ({B}, {C}, {H}, {W})")

    # 1-2. interpolating weight
    alpha = 0.0
    alpha = max(0.0, min(1.0, alpha))

    # 3-1. scalp texture and hair color interpolating logic
    T_orig_mixed = (1.0 - alpha) * T_list[0] + alpha * T_list[1]
    C_orig_mixed = (1.0 - alpha) * C_list[0] + alpha * C_list[1]

    # 3-3-1. Save interpolating result
    visualize_scalp_texture(T_orig_mixed, out_path, "scalp_texture_FINAL_MIXED")

    # 3-3-2. Save blending hair color result
    hair_color_dir = os.path.join(out_path, "intermediates", "hair_color")
    os.makedirs(hair_color_dir, exist_ok=True)
    save_color_map(C_orig_mixed, hair_color_dir, "hair_color_FINAL_MIXED")

    # 4. Blended Decoding Step
    # 4-1. Separate Scalp Texture (T) and Density Map (D)
    scalp_texture_mixed = T_orig_mixed[:, 0:-1, :, :] # latent code (T)
    density_map_mixed = T_orig_mixed[:, -1:, :, :]    # density map(D)

    # 4-2. Process the Density Map
    density_map_mixed = density_map_mixed * (0.5 / difflocks.model_config["sigma_data"]) + 0.5
    density_map_mixed = density_map_mixed.clamp(0, 1)
    density_map_mixed[density_map_mixed < 0.02] = 0.0

    return scalp_texture_mixed, density_map_mixed, C_orig_mixed


# --- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---
def Decode_to_3D_hair(
        scalp_texture_mixed,
        density_map_mixed,
        color_map_mixed,
        difflocks,
        out_path
):
    print("--- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---")

    # 1. Decode Scalp Texture to 3D Coordinates
    strand_points_world, strand_points_tbn, _root_uv = sample_strands_from_scalp_with_density(
        scalp_texture_mixed,
        density_map_mixed,
        difflocks.strand_codec,
        normalization_dict=difflocks.normalization_dict,
        scalp_mesh_data=difflocks.scalp_mesh_data,
        tbn_space_to_world_func=tbn_space_to_world,
        nr_chunks=difflocks.nr_chunks_decode_strands,
        upsample_multiplier=3)

    # 2. Output results
    npz_out_path=os.path.join(out_path, "mixed_output_strands.npz")
    save_dict = {"positions": strand_points_world.cpu().numpy()}
    if color_map_mixed is not None:
        # (1, 3, H, W) -> (H, W, 3)，供 npz2blender_kung.py 直接讀取上色
        save_dict["color_map"] = color_map_mixed.clamp(0, 1).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        print(f"Color Map shape: {save_dict['color_map'].shape} saved to NPZ.")
    np.savez(npz_out_path, **save_dict)
    print(f"save 3D hair strands: {npz_out_path}")

    return npz_out_path


# --- 4. Create blender file ---
def create_blender_file(
        args,
        out_path,
        npz_out_path
):  
    print("--- 4. Create blender file ---")

    args_mock = lambda: None
    args_mock.blender_path = args.blender_path
    args_mock.blender_nr_threads = 4
    args_mock.out_path = out_path
    args_mock.blender_strands_subsample = 1
    args_mock.blender_vertex_subsample = 1
    args_mock.alembic_resolution = 3
    args_mock.do_shrinkwrap = False
    args_mock.export_alembic = False

    # create cmd
    cmd=[
        args_mock.blender_path,
        "-t", str(args_mock.blender_nr_threads),
        "--background",
        "--python", os.path.join(ROOT, "inference", "npz2blender_kung.py"),
        "--",
        "--input_npz", npz_out_path,
        "--out_path", args_mock.out_path, 
        "--strands_subsample", str(args_mock.blender_strands_subsample), 
        "--vertex_subsample", str(args_mock.blender_vertex_subsample), 
        "--alembic_resolution", str(args_mock.alembic_resolution) 
    ]
    
    if args_mock.do_shrinkwrap:
        cmd.append("--shrinkwrap")
    if args_mock.export_alembic:
        cmd.append("--export_alembic")

    print("Import npz file into blender")    
    import subprocess
    subprocess.run(cmd, capture_output=False)
    print("Blender import script executed successfully.")


# 平移量delta y固定10 pixel
# def align_hairline_by_shifting(T_list, out_path, shift_pixels=10, soften_sigma=4.0, gap_blur_sigma=3.0, noise_std=0.03):
#     """
#     優化版：位移填補 + 邊緣軟混合 + 質感擾動
#     """
#     if not T_list or len(T_list) <= 1:
#         return T_list

#     B, C, H, W = T_list[0].shape
#     device = T_list[0].device

#     # 1. 取得聯集密度並羽化 (處理髮線與臉的過渡)
#     all_densities = torch.stack([T[:, -1:, :, :] for T in T_list])
#     D_union = all_densities.max(dim=0)[0] 
    
#     k_soft = int(soften_sigma * 4 + 1)
#     if k_soft % 2 == 0: k_soft += 1
#     D_union_soft = TF.gaussian_blur(D_union, [k_soft, k_soft], [soften_sigma, soften_sigma])

#     if out_path:
#         debug_dir = os.path.join(out_path, "debug_shifting_refined")
#         os.makedirs(debug_dir, exist_ok=True)
#         vutils.save_image(D_union_soft, os.path.join(debug_dir, "D_union_soft.png"))

#     aligned_list = []
#     for i, T_curr in enumerate(T_list):
#         feat = T_curr[:, :-1, :, :]
#         dens = T_curr[:, -1:, :, :]
        
#         # 2. 核心：位移填補
#         feat_shifted = torch.roll(feat, shifts=-shift_pixels, dims=2) 
        
#         # 3. 建立並「模糊」混合遮罩 (關鍵：消除突兀的直線切痕)
#         threshold = 0.01
#         gap_mask_raw = ((D_union > threshold).float() - (dens > threshold).float()).clamp(0, 1)
        
#         k_gap = int(gap_blur_sigma * 4 + 1)
#         if k_gap % 2 == 0: k_gap += 1
#         # 讓 gap_mask 有漸層，使填補內容與原圖內容無縫融合
#         gap_mask_soft = TF.gaussian_blur(gap_mask_raw, [k_gap, k_gap], [gap_blur_sigma, gap_blur_sigma])
        
#         # ===================================================
#         # 💡 在這裡加上輸出 gap_mask_soft 的程式碼
#         # ===================================================
#         if out_path:
#             # 建立一個專門放 gap_mask 的資料夾
#             gap_mask_dir = os.path.join(out_path, "debug_shifting_refined")
#             os.makedirs(gap_mask_dir, exist_ok=True)
#             # 呼叫你現有的黑白遮罩儲存函式，依據 index 命名（例如：gap_mask_soft_1.png）
#             save_grayscale_mask(gap_mask_soft, gap_mask_dir, f"gap_mask_soft_{i+1}")
#         # ===================================================

#         # 4. 質感擾動：在填補特徵中加入微量雜訊，防止解碼成「直髮」
#         # noise = torch.randn_like(feat_shifted) * noise_std
#         # feat_shifted_noisy = feat_shifted + noise
        
#         # 5. 利用軟邊緣進行特徵混合
#         feat_final = feat * (1.0 - gap_mask_soft) + feat_shifted * gap_mask_soft
#         # feat_final = feat * (1.0 - gap_mask_soft) + feat_shifted_noisy * gap_mask_soft
        
#         # 6. 讓所有特徵也跟著 D_union_soft 羽化，邊界才不會突兀
#         # feat_final = feat_final * D_union_soft
        
#         T_new = torch.cat([feat_final, D_union_soft], dim=1)
        
#         if out_path:
#             visualize_scalp_texture(T_new, debug_dir, f"refined_shift_texture_{i+1}")
            
#         aligned_list.append(T_new)
        
#     return aligned_list


# 非自適應平移量delta y固定10 pixel
def align_hairline_by_shifting_noadptive(T_list, out_path, shift_pixels=1, soften_sigma=0.0, gap_blur_sigma=3.0, noise_std=0.03):
    """
    優化版：位移填補 + 邊緣軟混合 + 質感擾動
    """
    if not T_list or len(T_list) <= 1:
        return T_list

    B, C, H, W = T_list[0].shape
    device = T_list[0].device

    # 1. 取得聯集密度並羽化 (處理髮線與臉的過渡)
    all_densities = torch.stack([T[:, -1:, :, :] for T in T_list])
    D_union = all_densities.max(dim=0)[0] 
    
    k_soft = int(soften_sigma * 4 + 1)
    if k_soft % 2 == 0: k_soft += 1
    D_union_soft = TF.gaussian_blur(D_union, [k_soft, k_soft], [soften_sigma, soften_sigma])

    if out_path:
        debug_dir = os.path.join(out_path, "debug_shifting_refined")
        os.makedirs(debug_dir, exist_ok=True)
        vutils.save_image(D_union_soft, os.path.join(debug_dir, "D_union_soft.png"))

    aligned_list = []
    for i, T_curr in enumerate(T_list):
        feat = T_curr[:, :-1, :, :]
        dens = T_curr[:, -1:, :, :]
        
        # 2. 核心：位移填補
        feat_shifted = torch.roll(feat, shifts=-shift_pixels, dims=2) 
        print(f"shift_pixels:{shift_pixels}")
        
        # 3. 建立並「模糊」混合遮罩 (關鍵：消除突兀的直線切痕)
        threshold = 0.01
        gap_mask_raw = ((D_union > threshold).float() - (dens > threshold).float()).clamp(0, 1)
        
        k_gap = int(gap_blur_sigma * 4 + 1)
        if k_gap % 2 == 0: k_gap += 1
        # 讓 gap_mask 有漸層，使填補內容與原圖內容無縫融合
        gap_mask_soft = TF.gaussian_blur(gap_mask_raw, [k_gap, k_gap], [gap_blur_sigma, gap_blur_sigma])
        
        # 4. 質感擾動：在填補特徵中加入微量雜訊，防止解碼成「直髮」
        noise = torch.randn_like(feat_shifted) * noise_std
        feat_shifted_noisy = feat_shifted + noise
        
        # 5. 利用軟邊緣進行特徵混合
        feat_final = feat * (1.0 - gap_mask_soft) + feat_shifted_noisy * gap_mask_soft
        
        # 6. 讓所有特徵也跟著 D_union_soft 羽化，邊界才不會突兀
        # feat_final = feat_final * D_union_soft
        
        T_new = torch.cat([feat_final, D_union_soft], dim=1)

        if out_path:
            visualize_scalp_texture(feat_shifted, debug_dir, f"shifted_only_{i+1}")
        
        if out_path:
            visualize_scalp_texture(T_new, debug_dir, f"refined_shift_texture_{i+1}")

        if out_path:
            # 建立一個專門放 gap_mask 的資料夾
            gap_mask_dir = os.path.join(out_path, "debug_shifting_refined")
            os.makedirs(gap_mask_dir, exist_ok=True)
            # 呼叫你現有的黑白遮罩儲存函式，依據 index 命名（例如：gap_mask_soft_1.png）
            save_grayscale_mask(gap_mask_soft, gap_mask_dir, f"gap_mask_soft_{i+1}")
            
        aligned_list.append(T_new)
        
    return aligned_list


# 自適應動態計算平移量delta y
def align_hairline_by_shifting(T_list, out_path, shift_pixels=10, soften_sigma=4.0, gap_blur_sigma=3.0, noise_std=0.08):
    """
    優化版：位移填補 + 邊緣軟混合 + 質感擾動
    """
    if not T_list or len(T_list) <= 1:
        return T_list

    B, C, H, W = T_list[0].shape
    device = T_list[0].device

    # 1. 取得聯集密度並羽化 (處理髮線與臉的過渡)
    all_densities = torch.stack([T[:, -1:, :, :] for T in T_list])
    D_union = all_densities.max(dim=0)[0] 
    
    k_soft = int(soften_sigma * 4 + 1)
    if k_soft % 2 == 0: k_soft += 1
    D_union_soft = TF.gaussian_blur(D_union, [k_soft, k_soft], [soften_sigma, soften_sigma])

    if out_path:
        debug_dir = os.path.join(out_path, "debug_shifting_refined")
        os.makedirs(debug_dir, exist_ok=True)
        vutils.save_image(D_union_soft, os.path.join(debug_dir, "D_union_soft.png"))

    aligned_list = []
    for i, T_curr in enumerate(T_list):
        feat = T_curr[:, :-1, :, :]
        dens = T_curr[:, -1:, :, :]
        
        # ---------------------------------------------------
        # 💡 Step A: 先計算出結構空缺 (Gap Mask)
        # ---------------------------------------------------
        threshold = 0.01
        gap_mask_raw = ((D_union > threshold).float() - (dens > threshold).float()).clamp(0, 1)
        
        # ---------------------------------------------------
        # 💡 Step B: 自適應計算位移量 (Adaptive Shift Calculation)
        # ---------------------------------------------------
        # gap_mask_raw 的形狀為 [B, 1, H, W]。我們沿著高度軸 (dim=2) 加總，
        # 這樣就能算出「每一個垂直行有幾個像素的破洞」。
        # 取 .max() 就能找到整張圖裡「最深的破洞」高度。
        max_gap_height = int(gap_mask_raw.sum(dim=2).max().item())
        
        # 加上安全邊距 (Margin) 確保能完全覆蓋，並設定上下限防止極端變形
        # margin = 5
        dynamic_shift = max_gap_height
        dynamic_shift = max(0, min(dynamic_shift, 30)) # 限制最大位移不超過 30 像素
        print(f"dynamic_shift:{dynamic_shift}")
        
        # ---------------------------------------------------
        # 💡 Step C: 核心位移填補與質感擾動 (現在使用動態位移量了！)
        # ---------------------------------------------------
        feat_shifted = torch.roll(feat, shifts=-dynamic_shift, dims=2) 

        if out_path:
            visualize_scalp_texture(feat_shifted, debug_dir, f"shifted_only_{i+1}")
        
        noise = torch.randn_like(feat_shifted) * noise_std
        # feat_shifted_noisy = feat_shifted + noise
        
        # ---------------------------------------------------
        # 💡 Step D: 邊緣羽化與特徵混合
        # ---------------------------------------------------
        k_gap = int(gap_blur_sigma * 4 + 1)
        if k_gap % 2 == 0: k_gap += 1
        gap_mask_soft = TF.gaussian_blur(gap_mask_raw, [k_gap, k_gap], [gap_blur_sigma, gap_blur_sigma])

        if out_path:
            # 建立一個專門放 gap_mask 的資料夾
            gap_mask_dir = os.path.join(out_path, "debug_shifting_refined")
            os.makedirs(gap_mask_dir, exist_ok=True)
            # 呼叫你現有的黑白遮罩儲存函式，依據 index 命名（例如：gap_mask_soft_1.png）
            save_grayscale_mask(gap_mask_soft, gap_mask_dir, f"gap_mask_soft_{i+1}")

        # if out_path:
        #     # 建立一個專門放 gap_mask 的資料夾
        #     gap_mask_dir = os.path.join(out_path, "debug_shifting_refined")
        #     os.makedirs(gap_mask_dir, exist_ok=True)
        #     # 呼叫你現有的黑白遮罩儲存函式，依據 index 命名（例如：gap_mask_soft_1.png）
        #     save_grayscale_mask(gap_mask_raw, gap_mask_dir, f"gap_mask_raw_{i+1}")

        # --- 方案一：空間變異雜訊注入 (取代原本的固定雜訊) ---
        # 產生與特徵圖同大小的隨機雜訊，確保每個像素點都有獨立的擾動
        # spatial_noise = torch.randn_like(feat_shifted) * noise_std
        # 雜訊強度與遮罩權重掛鉤，只在填補區生效，並隨機擾動
        # feat_shifted_noisy = feat_shifted + (spatial_noise * gap_mask_soft)
        # ---------------------------------------------------


        # 利用軟邊緣進行特徵混合
        # feat_final = feat * (1.0 - gap_mask_soft) + feat_shifted_noisy * gap_mask_soft
        feat_final = feat * (1.0 - gap_mask_soft) + (feat_shifted + noise) * gap_mask_soft
        # feat_final = feat * (1.0 - gap_mask_raw) + (feat_shifted + noise) * gap_mask_raw
        
        # 6. 讓所有特徵也跟著 D_union_soft 羽化，邊界才不會突兀
        # feat_final = feat_final * D_union_soft
        
        T_new = torch.cat([feat_final, D_union_soft], dim=1)
        
        if out_path:
            visualize_scalp_texture(T_new, debug_dir, f"refined_shift_texture_{i+1}")
            
        aligned_list.append(T_new)
        
    return aligned_list


#  加入delta x
def align_hairline_by_shifting_2D(T_list, out_path, soften_sigma=4.0, gap_blur_sigma=3.0, noise_std=0.03):
    if not T_list or len(T_list) <= 1:
        return T_list

    B, C, H, W = T_list[0].shape
    device = T_list[0].device

    # 1. 取得聯集密度並羽化
    all_densities = torch.stack([T[:, -1:, :, :] for T in T_list])
    D_union = all_densities.max(dim=0)[0] 
    
    k_soft = int(soften_sigma * 4 + 1)
    if k_soft % 2 == 0: k_soft += 1
    D_union_soft = TF.gaussian_blur(D_union, [k_soft, k_soft], [soften_sigma, soften_sigma])

    aligned_list = []
    for i, T_curr in enumerate(T_list):
        feat = T_curr[:, :-1, :, :]
        dens = T_curr[:, -1:, :, :]
        
        # 💡 Step A: 計算結構空缺
        threshold = 0.01
        gap_mask_raw = ((D_union > threshold).float() - (dens > threshold).float()).clamp(0, 1)
        
        # 💡 Step B: 自適應計算 2D 位移量 (dy, dx)
        # 縱向位移 (dy): 觀察每一列 (column) 的垂直積累
        max_gap_dy = int(gap_mask_raw.sum(dim=2).max().item())
        # 橫向位移 (dx): 觀察每一行 (row) 的水平積累
        max_gap_dx = int(gap_mask_raw.sum(dim=3).max().item())
        
        margin = 5
        dy = max(0, min(max_gap_dy + margin, 30))
        dx = max(0, min(max_gap_dx + margin, 20)) # 橫向通常不需要像縱向位移那麼多
        
        # 💡 Step C: 2D 張量滾動 (Tensor Roll)
        # 對 dims=2 (H) 位移 -dy, dims=3 (W) 位移 -dx
        feat_shifted = torch.roll(feat, shifts=(-dy, -dx), dims=(2, 3)) 
        
        noise = torch.randn_like(feat_shifted) * noise_std
        feat_shifted_noisy = feat_shifted + noise

        # 💡 Step D: 邊緣軟混合
        k_gap = int(gap_blur_sigma * 4 + 1)
        if k_gap % 2 == 0: k_gap += 1
        gap_mask_soft = TF.gaussian_blur(gap_mask_raw, [k_gap, k_gap], [gap_blur_sigma, gap_blur_sigma])
        
        feat_final = feat * (1.0 - gap_mask_soft) + feat_shifted_noisy * gap_mask_soft
        
        T_new = torch.cat([feat_final, D_union_soft], dim=1)
        aligned_list.append(T_new)
            
    return aligned_list


def align_hairline_by_shifting_hard(T_list, out_path, shift_pixels=10, soften_sigma=4.0, gap_blur_sigma=15.0, noise_std=0.03):
    """
    優化版：位移填補 + 邊緣軟混合 + 質感擾動
    """
    if not T_list or len(T_list) <= 1:
        return T_list

    B, C, H, W = T_list[0].shape
    device = T_list[0].device

    # 1. 取得聯集密度並羽化 (處理髮線與臉的過渡)
    all_densities = torch.stack([T[:, -1:, :, :] for T in T_list])
    D_union = all_densities.max(dim=0)[0] 
    
    k_soft = int(soften_sigma * 4 + 1)
    if k_soft % 2 == 0: k_soft += 1
    D_union_soft = TF.gaussian_blur(D_union, [k_soft, k_soft], [soften_sigma, soften_sigma])

    if out_path:
        debug_dir = os.path.join(out_path, "debug_shifting_refined")
        os.makedirs(debug_dir, exist_ok=True)
        vutils.save_image(D_union_soft, os.path.join(debug_dir, "D_union_soft.png"))

    aligned_list = []
    for i, T_curr in enumerate(T_list):
        feat = T_curr[:, :-1, :, :]
        dens = T_curr[:, -1:, :, :]
        
        # ---------------------------------------------------
        # 💡 Step A: 先計算出結構空缺 (Gap Mask)
        # ---------------------------------------------------
        threshold = 0.01
        gap_mask_raw = ((D_union > threshold).float() - (dens > threshold).float()).clamp(0, 1)
        
        # ---------------------------------------------------
        # 💡 Step B: 自適應計算位移量 (Adaptive Shift Calculation)
        # ---------------------------------------------------
        # gap_mask_raw 的形狀為 [B, 1, H, W]。我們沿著高度軸 (dim=2) 加總，
        # 這樣就能算出「每一個垂直行有幾個像素的破洞」。
        # 取 .max() 就能找到整張圖裡「最深的破洞」高度。
        max_gap_height = int(gap_mask_raw.sum(dim=2).max().item())
        
        # 加上安全邊距 (Margin) 確保能完全覆蓋，並設定上下限防止極端變形
        margin = 5
        dynamic_shift = max_gap_height + margin
        dynamic_shift = max(0, min(dynamic_shift, 30)) # 限制最大位移不超過 30 像素
        
        # ---------------------------------------------------
        # 💡 Step C: 核心位移填補與質感擾動 (現在使用動態位移量了！)
        # ---------------------------------------------------
        feat_shifted = torch.roll(feat, shifts=-dynamic_shift, dims=2) 
        
        noise = torch.randn_like(feat_shifted) * noise_std
        feat_shifted_noisy = feat_shifted + noise

       # ---------------------------------------------------
        # 💡 Step D: 邊緣羽化與特徵混合
        # ---------------------------------------------------
        k_gap = int(gap_blur_sigma * 4 + 1)
        if k_gap % 2 == 0: k_gap += 1
        
        # 取得高斯模糊後的軟遮罩 (用來萃取邊緣)
        gap_mask_soft = TF.gaussian_blur(gap_mask_raw, [k_gap, k_gap], [gap_blur_sigma, gap_blur_sigma])
        
        # === 【精確對照組：僅針對特徵邊緣進行 Feature-domain Smoothing】 ===
        
        # 1. 萃取「邊緣過渡帶」 (Edge Mask)
        # gap_mask_soft 的值介於 0 到 1 之間。交界處正好會是 0.5。
        # 透過數學轉換：1.0 - abs(值 - 0.5) * 2，我們可以得到一個「邊緣為 1，其他區域為 0」的遮罩。
        edge_mask = 1.0 - torch.abs(gap_mask_soft - 0.5) * 2.0
        
        # 2. 進行未經平滑的「硬拼接」
        feat_hard = feat * (1.0 - gap_mask_raw) + feat_shifted_noisy * gap_mask_raw
        
        # 3. 產生整張模糊的特徵
        feat_hard_blurred = TF.gaussian_blur(feat_hard, [k_gap, k_gap], [gap_blur_sigma, gap_blur_sigma])
        
        # 4. 終極融合：只在「邊緣過渡帶」套用模糊特徵，其餘保持硬拼接的高頻特徵
        feat_final = feat_hard * (1.0 - edge_mask) + feat_hard_blurred * edge_mask
        # ==========================================================

        # 6. 讓所有特徵也跟著 D_union_soft 羽化，邊界才不會突兀
        # feat_final = feat_final * D_union_soft
        
        T_new = torch.cat([feat_final, D_union_soft], dim=1)
        
        if out_path:
            visualize_scalp_texture(T_new, debug_dir, f"refined_shift_texture_{i+1}")
            
        aligned_list.append(T_new)
        
    return aligned_list


def run_kung():

    parser = argparse.ArgumentParser(description="Kung Hair Inference")
    parser.add_argument("--input_config", type=str, default=None,
                         help="JSON file with 'input_file_paths' and 'mask_file_paths'")
    parser.add_argument("--blender_path", type=str, default=BLENDER_PATH)
    parser.add_argument("--output_path", type=str, default="./outputs_inference/")
    args = parser.parse_args()

    # --- input model setting ---
    path_strand_codec=os.path.join(ROOT, "./checkpoints/strand_vae/strand_codec.pt")
    path_config = os.path.join(ROOT, "./configs/config_scalp_texture_conditional.json")
    path_diffusion_model_ckpt_path = os.path.join(ROOT, "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth") #longest trained one yet
    path_material_model_ckpt_path = os.path.join(ROOT, "./checkpoints/rgb2material/rgb2material.pt")


    # --- output result setting ---
    out_path=args.output_path
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "masks"), exist_ok=True)


    # --- input file setting (預設值，會被 --input_config 覆蓋) ----
    input_file_paths = [
        "./samples/hair/freeman_2.png",   # base
        "./samples/hair/uggams_3.png",
        "./samples/hair/cooper_4.jpg",
        "./samples/hair/hathaway_1.jpg",
    ]
    mask_file_paths = [
        "./samples/mask/mask1.png",
        "./samples/mask/mask2.png",
        "./samples/mask/mask3.png",
    ]

    if args.input_config and os.path.exists(args.input_config):
        with open(args.input_config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        input_file_paths = cfg.get("input_file_paths", input_file_paths)
        mask_file_paths = cfg.get("mask_file_paths", mask_file_paths)
        print(f"Loaded input_config: {args.input_config} "
              f"({len(input_file_paths)} images, {len(mask_file_paths)} masks)")
    elif args.input_config:
        print(f"警告: 找不到 --input_config 指定的檔案 {args.input_config}，使用預設輸入清單。")

    # --- 1. Generate multiple Scalp Texture ---
    difflocks, T_list, C_list = scalp_texture(
        path_strand_codec,
        path_config,
        path_diffusion_model_ckpt_path,
        path_material_model_ckpt_path,
        input_file_paths,
        out_path,
        mask_file_paths
    )

    # ===================================================
    # 💡 在這裡提取「剛經 Diffusion 生成、完全未經過任何對齊與混合」的原始密度圖
    # ===================================================
    if out_path:
        raw_density_dir = os.path.join(out_path, "intermediates", "density")
        os.makedirs(raw_density_dir, exist_ok=True)
        
        for i, T_raw in enumerate(T_list):
            # 1. 提取第 65 個通道 (最後一個 channel)
            raw_density = T_raw[:, -1:, :, :]
            
            # 2. 執行 Diffusion 的反量化公式，轉回 0~1 的可視化區間
            raw_density = raw_density * (0.5 / difflocks.model_config["sigma_data"]) + 0.5
            raw_density = raw_density.clamp(0, 1)
            
            # 3. 儲存圖片 (會產生 raw_density_1.png, raw_density_2.png ...)
            save_grayscale_mask(raw_density, raw_density_dir, f"density_{i+1}")
    # ===================================================

    # --- 1.5 Align Hairline ---
    T_list = align_hairline_by_shifting(
        T_list, 
        out_path, 
        shift_pixels=20, 
        soften_sigma=3.0,
        noise_std=0.08
    )

    # T_list = align_hairline_by_shifting_noadptive(
    #     T_list, 
    #     out_path, 
    #     shift_pixels=10, 
    #     soften_sigma=3.0,
    #     noise_std=0.08
    # )
    # T_list = align_hairline_by_shifting_hard(
    #     T_list, 
    #     out_path, 
    #     shift_pixels=20, 
    #     soften_sigma=4.0,
    #     noise_std=0.05
    # )
    # T_list = align_hairline_by_shifting_2D(
    #     T_list, 
    #     out_path, 
    #     # shift_pixels=20, 
    #     soften_sigma=4.0,
    #     noise_std=0.05
    # )
    
    

    print(f"DEBUG: T_list 數量 = {len(T_list)}, C_list 數量 = {len(C_list)}")

    # --- 2-1. Executing the Scalp Texture Blending Logic ---
    scalp_texture_mixed, density_map_mixed, color_map_mixed = scalp_texture_blending(
            T_list,
            C_list,
            mask_file_paths,
            out_path,
            difflocks
    )
    # scalp_texture_mixed, density_map_mixed, color_map_mixed = scalp_texture_blending_traditional(
    #         T_list,
    #         C_list,
    #         mask_file_paths,
    #         out_path,
    #         difflocks
    # )

    # --- 2-2. Executing the Scalp Texture interpolating Logic ---
    # scalp_texture_mixed, density_map_mixed = scalp_texture_interpolating(
    #         T_list,
    #         C_list,
    #         mask_file_paths,
    #         out_path,
    #         difflocks
    # )

    # # ==================== 讀取指定圖片 ====================
    # # 💡 在這裡直接 import cv2，可以完美避開跟函數其他地方的同名衝突
    # # import cv2 
    # # import torch
    
    # target_img_path = "./samples/hair_color/hc10.jpg"
    # print(f"正在從讀取指定的顏色圖片: {target_img_path}")
    
    # if not os.path.exists(target_img_path):
    #     raise FileNotFoundError(f"找不到路徑：{target_img_path}")
        
    # # 這裡就不會再噴 UnboundLocalError 了
    # img_bgr = cv2.imread(target_img_path) 
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # device = scalp_texture_mixed.device
    # color_map_mixed = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # color_map_mixed /= 255.0
    # # ===================================================

    
    # --- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---
    npz_out_path = Decode_to_3D_hair(
            scalp_texture_mixed,
            density_map_mixed,
            color_map_mixed,
            difflocks,
            out_path
    )
    
    # --- 4. Create .blend ---
    create_blender_file(
        args,
        out_path,
        npz_out_path
    )



# test hair color
def test_create_blender_file():

    # path setting
    out_path="./outputs_inference/"
    npz_out_path="./outputs_inference/mixed_output_strands.npz"

    # --- blender path setting ---
    class Args:
        def __init__(self):
            self.blender_path = BLENDER_PATH 
            
    try:
        args
    except NameError:
        args = Args()

    # --- 4. Create .blend ---
    create_blender_file(
        args,
        out_path,
        npz_out_path
    )



#=====================================================================================
#======================== Revision Region End ========================================
#=====================================================================================


if __name__ == '__main__':

    # run()
    run_kung()
    # test_create_blender_file()