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
from utils.diffusion_utils import multi_diffusion,  multi_diffusion_cfg, multi_diffusion_cfg_v2

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
    def extract_dinov2_features(self, rgb_img):
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

        extra_args = {}
        extra_args = {
            'latents_dict': {
                "dinov2": {
                    "cls_token": cls_tok,
                    "final_latent": patch_embeddings_reshaped,
                }
            }
        }
        
        return extra_args
    

    
    def generate_mixed_texture(self, img_list, mask_list):
        all_extra_args = []

        for img in img_list:
            all_extra_args.append(self.extract_dinov2_features(img))
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mixed_texture = multi_diffusion_cfg_v2(
                model_ema=self.model_ema,
                model_config=self.model_config,
                multi_extra_args=all_extra_args, # feature list
                masks=mask_list,                 # mask list
                nr_iters=self.nr_iters_denoise
            )

        return mixed_texture.float()



    # def _generate_texture_from_rgb(self, rgb_img):
        
    #     extra_args={}
    #     extra_args['latents_dict']={}
        
    #     # dinov2 v2 
    #     rgb_input = self.dinov2_latents_preprocessor(rgb_img).to("cuda")
    #     dinov2_output = self.dinov2_latents_model.forward_features(rgb_input)
    #     patch_tok = dinov2_output["x_norm_patchtokens"].clone()
    #     cls_tok = dinov2_output["x_norm_clstoken"].clone()
    #     cls_token=cls_tok
    #     patch_embeddings = patch_tok
        
    #     # Reshape patch tokens
    #     batch_size, num_patches, hidden_size = patch_embeddings.shape
    #     h = w = int(num_patches ** 0.5)
    #     patch_embeddings_reshaped = patch_embeddings.reshape(batch_size, h, w, hidden_size)
    #     patch_embeddings_reshaped=patch_embeddings_reshaped.permute(0,3,1,2).contiguous() #Make it bchw 
        
    #     extra_args['latents_dict']["dinov2"]={
    #                                 "cls_token": cls_token, # <--- Global Cond
    #                                 "final_latent": patch_embeddings_reshaped,
    #                                 }
        
    #     # run diffusion
    #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #         scalp_texture_orig = sample_images_cfg(
    #             1, 
    #             cfg_val=self.cfg_val, 
    #             cfg_interval=[0.0, 5.0], 
    #             model_ema=self.model_ema, 
    #             model_config=self.model_config, 
    #             nr_iters=self.nr_iters_denoise, 
    #             extra_args=extra_args)
    #     scalp_texture_orig = scalp_texture_orig.float()
        
    #     return scalp_texture_orig, cls_token
    

    def extract_hair_color(self, frame_cropped, out_path, idx):
            
        # 2. 取得頭髮遮罩 (與 frame_cropped 完全對齊)
        # 注意：在 selfie_multiclass 模型中，類別 1 通常是頭髮
        category_mask = self.mediapipe_img.run_segmentation(frame_cropped)
        hair_mask_2d = (category_mask == 1) 

        # 3. 提取顏色
        hair_pixels = frame_cropped[hair_mask_2d]
        
        if len(hair_pixels) > 0:
            # 取中位數 RGB
            extracted_rgb = np.median(hair_pixels, axis=0).astype(np.uint8)
        else:
            extracted_rgb = np.array([50, 50, 50], dtype=np.uint8) # 預設深灰色

        # 4. 生成 256x256 顏色圖片並儲存
        if out_path:
            os.makedirs(out_path, exist_ok=True)
            
            # 建立純色圖
            color_img_256 = np.zeros((256, 256, 3), dtype=np.uint8)
            color_img_256[:] = extracted_rgb
            
            # 儲存 (RGB -> BGR)
            # color_save_path = os.path.join(out_path, f"hair_color_pure_{idx}.png")
            # cv2.imwrite(color_save_path, cv2.cvtColor(color_img_256, cv2.COLOR_RGB2BGR))
            if out_path:
                # 1. 定義頭髮顏色專用的子資料夾路徑
                hair_color_dir = os.path.join(out_path, "hair_color")
                hair_mask_dir = os.path.join(out_path, "hair_mask")
                
                # 2. 如果資料夾不存在，就自動生成 (exist_ok=True 防止重複報錯)
                os.makedirs(hair_color_dir, exist_ok=True)
                os.makedirs(hair_mask_dir, exist_ok=True)
                
                # 3. 更新圖片儲存路徑，指向該子資料夾
                color_save_path = os.path.join(hair_color_dir, f"hair_color_pure_{idx}.png")
                
                # 建立純色圖 (256x256)
                color_img_256 = np.zeros((256, 256, 3), dtype=np.uint8)
                color_img_256[:] = extracted_rgb
                
                # 儲存圖片 (RGB -> BGR)
                cv2.imwrite(color_save_path, cv2.cvtColor(color_img_256, cv2.COLOR_RGB2BGR))
                
                # (選配) 若要儲存遮罩，建議也放在子資料夾或另外建立
                mask_save_path = os.path.join(hair_mask_dir, f"hair_mask_{idx}.png")
                cv2.imwrite(mask_save_path, (hair_mask_2d * 255).astype(np.uint8))
                
                print(f"成功提取顏色: {extracted_rgb}, 已儲存至 {color_save_path}")
    

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

            # extract hair color
            self.extract_hair_color(frame_cropped, out_path, idx)

            # Back to tensor for DINOv2
            rgb_img_cropped = torch.tensor(frame_cropped).cuda()
            rgb_img_cropped = rgb_img_cropped.permute(2,0,1).unsqueeze(0).float()/255.0
            
            
            # Generate Scalp Texture
            scalp_texture_orig, cls_token = self._generate_texture_from_rgb(rgb_img_cropped)

            
            all_texture_data.append({
                "scalp_texture_orig": scalp_texture_orig,
                "cls_token": cls_token,
                "input_file": file_path
            })
            
            if out_path:
                 os.makedirs(out_path, exist_ok=True)
                 np.savez(os.path.join(out_path, f"scalp_texture_{idx}.npz"), scalp_texture=scalp_texture_orig.cpu().numpy())

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

    path_strand_codec="./checkpoints/strand_vae/strand_codec.pt"
    path_config = "./configs/config_scalp_texture_conditional.json"
    path_diffusion_model_ckpt_path = "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth" #longest trained one yet
    path_material_model_ckpt_path = "./checkpoints/rgb2material/rgb2material.pt" 
    
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

    for i, T_orig in enumerate(T_list):
        visualize_scalp_texture(T_orig, out_path, f"scalp_texture_{i+1}")
    
    # print (f"T_list: {len(T_list)}")
    # print (f"mask: {len(mask_file_paths)}")
    if len(T_list) != len(mask_file_paths) + 1:
        print(f"Error: Input images ({len(T_list)}) "
              "must be one more than masks ({len(mask_file_paths)}).")
        return

    return difflocks, T_list, 


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

    # 2-1. Turn C-list(NumPy) into Tensor
    # converted sharp: (Batch, 3, 256, 256), range: [0, 1]
    C_tensors = []
    for c_img in C_list:
        c_tensor = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        C_tensors.append(c_tensor)

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
    C_orig_mixed = C_tensors[0].clone()
    
    # 3-2. Iterative Loop : M1/T2, M2/T3, M3/T4 ...
    for i in range(len(M_list_T)):
        M_T = M_list_T[i] 
        M_C = M_list_C[i]         
        T_target = T_list[i+1] 
        C_target = C_tensors[i+1]

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
    c_final_np = (C_orig_mixed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(hair_color_dir, "hair_color_FINAL_MIXED.png"), cv2.cvtColor(c_final_np, cv2.COLOR_RGB2BGR))

    # 4. Blended Decoding Step
    # 4-1. Separate Scalp Texture (T) and Density Map (D)
    scalp_texture_mixed = T_orig_mixed[:, 0:-1, :, :] # latent code (T)
    density_map_mixed = T_orig_mixed[:, -1:, :, :]    # density map(D)
    
    # 4-2. Process the Density Map
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

    # 2. Turn C-list(NumPy) into Tensor
    # converted sharp: (Batch, 3, 256, 256), range: [0, 1]
    C_tensors = []
    for c_img in C_list:
        c_tensor = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        C_tensors.append(c_tensor)

    # 3-1. scalp texture and hair color interpolating logic
    T_orig_mixed = T_list[0].clone()
    C_orig_mixed = C_tensors[0].clone()

    T_orig_mixed = (1.0 - alpha) * T_list[0] + alpha * T_list[1]
    C_orig_mixed = (1.0 - alpha) * C_tensors[0] + alpha * C_tensors[1]

    # 3-3-1. Save interpolating result
    visualize_scalp_texture(T_orig_mixed, out_path, "scalp_texture_FINAL_MIXED")
    # print("Finish Multiple Scalp Texture Blending")

    # 3-3-2. Save blending hair color result
    hair_color_dir = os.path.join(out_path, "intermediates", "hair_color")
    os.makedirs(hair_color_dir, exist_ok=True)
    c_final_np = (C_orig_mixed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(hair_color_dir, "hair_color_FINAL_MIXED.png"), cv2.cvtColor(c_final_np, cv2.COLOR_RGB2BGR))

    # 4. Blended Decoding Step
    # 4-1. Separate Scalp Texture (T) and Density Map (D)
    scalp_texture_mixed = T_orig_mixed[:, 0:-1, :, :] # latent code (T)
    density_map_mixed = T_orig_mixed[:, -1:, :, :]    # density map(D)
    
    # 4-2. Process the Density Map
    density_map_mixed = density_map_mixed * (0.5 / difflocks.model_config["sigma_data"]) + 0.5 
    density_map_mixed = density_map_mixed.clamp(0, 1)
    density_map_mixed[density_map_mixed < 0.02] = 0.0 

    return scalp_texture_mixed, density_map_mixed


# --- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---
def Decode_to_3D_hair(
        scalp_texture_mixed,
        density_map_mixed,
        difflocks,
        out_path
):
    print("--- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---")
    
    # 1. Decode Scalp Texture to 3D Coordinates
    strand_points_world, strand_points_tbn = sample_strands_from_scalp_with_density(
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
    np.savez(npz_out_path, positions=strand_points_world.cpu().numpy())
    print(f"save 3D hair strands: {npz_out_path}")

    return npz_out_path

def get_colors_from_uv(color_map_mixed, root_uv01):
    # 1. 轉換至 -1 ~ 1 空間
    root_uv11 = root_uv01 * 2.0 - 1.0
    
    # 2. 座標提取
    # 假設 root_uv01 原始順序是 [y, x]
    y_raw = root_uv11[:, 0]
    x_raw = root_uv11[:, 1]
    
    # 【最終修正方案】：
    # 如果灰白色（左下）跑到了左上，代表 Y 軸需要從「反向」切換回「正向」
    # 嘗試移除負號，直接使用原始 y 比例
    y_corrected = y_raw 
    x_corrected = x_raw 
    
    # 3. 建立採樣網格 (必須是 [x, y] 順序)
    sampling_grid = torch.stack([x_corrected, y_corrected], dim=-1).view(1, 1, -1, 2)
    
    # 4. 執行採樣
    sampled_colors = torch.nn.functional.grid_sample(
        color_map_mixed, 
        sampling_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )
    
    return sampled_colors.squeeze().permute(1, 0)


# def Decode_to_3D_hair(
#         scalp_texture_mixed,
#         density_map_mixed,
#         color_map_mixed,
#         difflocks,
#         out_path
# ):
#     print("--- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---")
    
#     # 1. Decode Scalp Texture to 3D Coordinates
#     strand_points_world, strand_points_tbn, root_uv = sample_strands_from_scalp_with_density(
#         scalp_texture_mixed, 
#         density_map_mixed, 
#         difflocks.strand_codec, 
#         normalization_dict=difflocks.normalization_dict, 
#         scalp_mesh_data=difflocks.scalp_mesh_data, 
#         tbn_space_to_world_func=tbn_space_to_world, 
#         nr_chunks=difflocks.nr_chunks_decode_strands, 
#         upsample_multiplier=3)
    
#     # 2. 獲取對應的顏色 (假設採樣順序與解碼順序一致)
#     # 如果你的 sample 函數內包含隨機性，建議在 sample 函數內部捕捉 uv
#     root_colors = get_colors_from_uv(color_map_mixed, root_uv)

#     # 2. Output results
#     npz_out_path=os.path.join(out_path, "mixed_output_strands.npz")
#     np.savez(npz_out_path, 
#              positions=strand_points_world.cpu().numpy(),
#              colors=root_colors.cpu().numpy())
#     print(f"save 3D hair strands: {npz_out_path}")

#     return npz_out_path


# --- 4. Create blender file ---
def create_blender_file(
        args,
        out_path,
        npz_out_path
):  
    print("--- 4. Create blender file ---")

    if args.blender_path != "": 
        args_mock = lambda: None
        args_mock.blender_path = "/home/kyh/blender/blender" 
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
        "--python", "./inference/npz2blender_kung2.py",
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



def align_hairline_by_shifting(T_list, out_path, shift_pixels=10, soften_sigma=4.0, gap_blur_sigma=3.0, noise_std=0.03):
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
            visualize_scalp_texture(T_new, debug_dir, f"refined_shift_texture_{i+1}")
            
        aligned_list.append(T_new)
        
    return aligned_list

    

def run_kung():

    # --- blender path setting ---
    class Args:
        def __init__(self):
            self.blender_path = "/home/kyh/blender/blender" 
            
    try:
        args
    except NameError:
        args = Args()

    # --- input model setting ---
    path_strand_codec="./checkpoints/strand_vae/strand_codec.pt"
    path_config = "./configs/config_scalp_texture_conditional.json"
    path_diffusion_model_ckpt_path = "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth" #longest trained one yet
    path_material_model_ckpt_path = "./checkpoints/rgb2material/rgb2material.pt" 
    

    # --- output result setting ---
    out_path="./outputs_inference/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "masks"), exist_ok=True)


    # --- input file setting ----
    input_file_paths = [
        "./samples/hair/freeman_2.png",   # base
        "./samples/hair/uggams_3.png",
        "./samples/hair/cooper_4.jpg",
        "./samples/hair/hathaway_1.jpg"
        # "./samples/hair/21.png",
        # "./samples/hair/buzzcut_3.jpg",
        # "./samples/hair/16.png"
    ]
    
    mask_file_paths = [
        "./samples/mask/M0/mask1.png",  
        "./samples/mask/M0/mask2.png",  
        "./samples/mask/M0/mask3.png"
        # "./samples/mask/M7/M73.png",
        # "./samples/mask/M7/M74.png"
    ]

    # --- 1. Initialize Model ---
    difflocks = DiffLocksInference(
        path_strand_codec, 
        path_config, 
        path_diffusion_model_ckpt_path, 
        path_material_model_ckpt_path
    )
    

    # --- 2. Prepare all image feature and masks ---
    input_img_tensors = []
    processed_masks = []
    H, W = difflocks.model_config['input_size'] # 通常是 256x256

    # 2.1 遍歷圖片路徑並提取 Tensor
    for i, img_path in enumerate(input_file_paths):
        frame = cv2.imread(img_path)

        # --- 新增這段檢查 ---
        if frame is None:
            print(f"找不到圖片或讀取失敗！路徑：{img_path}")
            continue # 跳過這張圖，繼續處理下一張
        # ------------------

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用 Mediapipe 裁切臉部
        _, face_landmarks = difflocks.mediapipe_img.run(frame)
        if face_landmarks is None: continue
        frame_cropped = crop_face(frame, face_landmarks, output_size=770)
        
        # 轉成 Tensor 並存入列表
        img_t = torch.tensor(frame_cropped).cuda().permute(2,0,1).unsqueeze(0).float()/255.0
        input_img_tensors.append(img_t)

        # 2.2 準備對應遮罩 (MultiDiffusion 邏輯)
        if i == 0:
            # 第一張為 Base，初始為全白 (1.0)
            base_mask = torch.ones((1, 1, H, W)).cuda()
            processed_masks.append(base_mask)
        else:
            # 載入並縮放遮罩 (M1 對應 T2, M2 對應 T3...)
            m_path = mask_file_paths[i-1]
            mask_t = load_and_resize_mask(m_path, H, W, "cuda", blur_sigma=9.0)
            processed_masks.append(mask_t)
            
            # 從 Base 中扣除當前遮罩，確保區域不重疊過多
            processed_masks[0] = (processed_masks[0] - mask_t).clamp(0, 1)

    
    # --- 3. 執行多重擴散融合生成 (取代舊的階段 1 與 2) ---
    print("--- 3. Executing Latent Space MultiDiffusion ---")
    # 直接呼叫你定義好的 generate_mixed_texture
    scalp_texture_mixed_raw = difflocks.generate_mixed_texture(
        img_list=input_img_tensors, 
        mask_list=processed_masks
    )

    visualize_scalp_texture(
        scalp_texture_latent=scalp_texture_mixed_raw, 
        output_path=out_path, 
        filename="FINAL_MIXED_SCALP_TEXTURE"
    )

    # 分離 Texture 與 Density 並處理密度 (邏輯保持不變)
    scalp_texture_mixed = scalp_texture_mixed_raw[:, 0:-1, :, :]
    density_map_raw = scalp_texture_mixed_raw[:, -1:, :, :]
    
    density_map_mixed = density_map_raw * (0.5 / difflocks.model_config["sigma_data"]) + 0.5 
    density_map_mixed = density_map_mixed.clamp(0, 1)
    density_map_mixed[density_map_mixed < 0.02] = 0.0




    



    # --- 1. Generate multiple Scalp Texture ---
    # difflocks, T_list = scalp_texture(
    #     path_strand_codec,
    #     path_config,
    #     path_diffusion_model_ckpt_path,
    #     path_material_model_ckpt_path,
    #     input_file_paths,
    #     out_path,
    #     mask_file_paths
    # )

    # --- 1.5 Align Hairline ---
    # T_list = align_hairline_by_shifting(
    #     T_list, 
    #     out_path, 
    #     shift_pixels=20, 
    #     soften_sigma=4.0,
    #     noise_std=0.05
    # )

    # # --- 1.5 Extract Haircolor ---
    # # read C_list
    # # 1. 確保路徑指向 intermediates/hair_color
    # hair_color_dir = os.path.join(out_path, "intermediates", "hair_color")
    
    # # 2. 讀取並嚴格排序 (確保 0, 1, 2, 3 的順序正確)
    # color_files = sorted(
    #     [f for f in os.listdir(hair_color_dir) if f.startswith("hair_color_pure_") and f.endswith(".png")],
    #     key=lambda x: int(x.split('_')[-1].split('.')[0])
    # )
    
    # C_list = []
    # for filename in color_files:
    #     img_path = os.path.join(hair_color_dir, filename)
    #     img_bgr = cv2.imread(img_path)
    #     if img_bgr is not None:
    #         # 轉為 RGB 以符合後續 Tensor 運算
    #         C_list.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
    # # 3. 除錯檢查：這行非常重要，能幫你確認數量
    # print(f"DEBUG: T_list 數量 = {len(T_list)}, C_list 數量 = {len(C_list)}")
    
    # if len(C_list) < len(T_list):
    #     raise ValueError(f"錯誤：髮色圖片數量({len(C_list)}) 少於紋理數量({len(T_list)})！請檢查資料夾。")
    

    # # --- 2-1. Executing the Scalp Texture Blending Logic ---
    # scalp_texture_mixed, density_map_mixed, color_map_mixed = scalp_texture_blending(
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


    # --- 3. Decode the Blended Scalp Texture to 3D Hair Strands ---
    npz_out_path = Decode_to_3D_hair(
            scalp_texture_mixed,
            density_map_mixed,
            # color_map_mixed,
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
            self.blender_path = "/home/kyh/blender/blender" 
            
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