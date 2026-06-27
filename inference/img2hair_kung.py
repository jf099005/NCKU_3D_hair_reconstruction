#!/usr/bin/env python3


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

torch.autograd.set_grad_enabled(False)



class Mediapipe():
    def __init__(self, mode):
        super(Mediapipe, self).__init__()

        self.mode=mode

        base_options = python.BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR,'./assets/face_landmarker.task'),
            delegate=mp.tasks.BaseOptions.Delegate.GPU
            )
        options = vision.FaceLandmarkerOptions(
                                        running_mode=mode,
                                        base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=10,
                                        min_face_detection_confidence=0.1,
                                        min_face_presence_confidence=0.1,
                                        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def run(self, rgb_image_numpy):
    
        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_numpy)


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
            img_to_process = frame
            rgb_img_tensor = torch.tensor(img_to_process).cuda()
            rgb_img_tensor = rgb_img_tensor.permute(2,0,1).unsqueeze(0).float()/255.0
            
            frame_mp=(rgb_img_tensor.permute(0,2,3,1).squeeze(0)*255.0).to(torch.uint8) 
            frame_mp=frame_mp.detach().cpu().numpy()
            face_landmarks_px, face_landmarks = self.mediapipe_img.run(frame_mp)
            if face_landmarks is None: 
                print(f"No face detected in {file_path}. Skipping.")
                continue
            frame_cropped=crop_face(frame_mp, face_landmarks, output_size=770)

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


def load_and_resize_mask(mask_path: str, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    """
    載入遮罩圖片，轉為 PyTorch 張量，並縮放到目標尺寸。
    
    Args:
        mask_path: 遮罩圖片檔案路徑。
        target_h, target_w: Scalp Texture 的高度和寬度 (例如 256, 256)。
        device: 張量將被移動到的設備 (例如 'cuda')。
        
    Returns:
        一個形狀為 (1, 1, target_h, target_w) 的浮點數張量，值在 [0, 1]。
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"找不到遮罩檔案: {mask_path}")
        
    # 1. 讀取圖片：以灰度模式讀取 (確保只有一個通道)
    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_np is None:
        raise ValueError(f"無法讀取遮罩圖片: {mask_path}")
        
    # 2. 轉換為 PyTorch 張量：HWC -> CHW (1, H, W)
    mask_tensor = torch.from_numpy(mask_np).float().to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    # 3. 正規化：將 0-255 範圍轉為 0.0-1.0 範圍
    mask_tensor /= 255.0
    
    # 4. 縮放：縮放到 Scalp Texture 的尺寸 (例如 256x256)
    # 使用 bilinear 插值來保持柔和邊緣
    resized_mask = F.interpolate(
        mask_tensor, 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    )
    
    # 確保遮罩在 0 到 1 之間
    return resized_mask.clamp(0, 1)


def run_kung():

    class Args:
        def __init__(self):
            self.blender_path = "/home/kyh/blender/blender" 
            
    try:
        args
    except NameError:
        args = Args()

    path_strand_codec="./checkpoints/strand_vae/strand_codec.pt"
    path_config = "./configs/config_scalp_texture_conditional.json"
    path_diffusion_model_ckpt_path = "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth" #longest trained one yet
    path_material_model_ckpt_path = "./checkpoints/rgb2material/rgb2material.pt" 
    
    out_path="./outputs_inference/"
    os.makedirs(out_path, exist_ok=True)

    #run----
    img_path1="./samples/hathaway_1.jpg"
    img_path2="./samples/uggams_3.png"
    input_file_paths = [img_path1, img_path2]

    difflocks= DiffLocksInference(
        path_strand_codec, 
        path_config, 
        path_diffusion_model_ckpt_path, 
        path_material_model_ckpt_path)
    print("--- 1. Generate multiple Scalp Texture ---")

    all_texture_data = difflocks.files_to_scalp_textures(
        file_paths=input_file_paths, 
        out_path=os.path.join(out_path, "intermediates")
    )
    
    if len(all_texture_data) < 2:
        print("Error：At least two images are required for blending.")
        return

    T_orig_A = all_texture_data[0]['scalp_texture_orig'] 
    T_orig_B = all_texture_data[1]['scalp_texture_orig'] 
    visualize_scalp_texture(T_orig_A, out_path, "scalp_texture_A_orig")
    visualize_scalp_texture(T_orig_B, out_path, "scalp_texture_B_orig")
    
    print(f"T_A shape: {T_orig_A.shape}, T_B shape: {T_orig_B.shape}")
    
    print("\n--- 2. 執行 Scalp Texture 混合邏輯 ---")
    
    # === 您的 Scalp Texture 混合邏輯（替換為自定義圖片遮罩）===

    # 1. 定義自定義遮罩圖片路徑
    mask_path = "./samples/mask/mask1.png" # <--- 請將此路徑替換為您的圖片路徑

    # 2. 獲取尺寸和設備
    B, C, H, W = T_orig_A.shape
    device = T_orig_A.device

    # 3. 載入並處理自定義遮罩 M
    try:
        # 載入遮罩。輸出的 mask_M 尺寸應為 (1, 1, H, W)
        mask_M = load_and_resize_mask(mask_path, H, W, device)
    except (FileNotFoundError, ValueError) as e:
        print(f"遮罩載入錯誤：{e}。將使用矩形預設遮罩。")
        # 如果載入失敗，您可以選擇 fallback 到矩形遮罩，或直接退出
        # 這裡為演示，假設它會成功。
        raise e


    # 4. **視覺化遮罩** (使用之前定義的 save_grayscale_mask 函式)
    # T_orig_mixed = T_orig_A.clone() 
    # save_grayscale_mask(mask_M, out_path, "mask_custom_input")


    # 5. 執行線性混合 (使用遮罩 M 來混合 T_A 和 T_B)
    # T_orig_mixed = (1 - M) * T_A + M * T_B
    # 遮罩 M 的尺寸是 (1, 1, H, W)，T_A/T_B 尺寸是 (B, C, H, W)
    # PyTorch 的廣播 (Broadcasting) 機制會自動複製 M 到 C 個通道上

    # 擴展遮罩到所有通道 (從 1 到 C)
    mask_M_expanded = mask_M.expand_as(T_orig_A) 

    # 線性混合 Scalp Texture (T) 和密度圖 (D)
    T_orig_mixed = (1.0 - mask_M_expanded) * T_orig_A + mask_M_expanded * T_orig_B


    # 視覺化混合後的 Scalp Texture
    visualize_scalp_texture(T_orig_mixed, out_path, "scalp_texture_MIXED")

    print("Scalp Texture 已使用自定義圖片遮罩完成混合。")

    # === 混合後的解碼步驟 ===
    
    # 步驟 4: 分離 Scalp Texture (T) 和密度圖 (D)
    scalp_texture_mixed = T_orig_mixed[:, 0:-1, :, :] # 潛在碼 (T)
    density_map_mixed = T_orig_mixed[:, -1:, :, :]   # 密度圖 (D)
    
    # 步驟 5: 處理密度圖 (與原程式碼中的邏輯一致)
    density_map_mixed = density_map_mixed * (0.5 / difflocks.model_config["sigma_data"]) + 0.5 
    density_map_mixed = density_map_mixed.clamp(0, 1)
    density_map_mixed[density_map_mixed < 0.02] = 0.0 
    
    print("\n--- 3. 解碼混合後的 Scalp Texture 到 3D 髮束 ---")
    
    # 步驟 6: 解碼 Scalp Texture 到 3D 座標
    # 注意: 這裡需要確保 sample_strands_from_scalp_with_density 函式在作用域內
    strand_points_world, strand_points_tbn = sample_strands_from_scalp_with_density(
        scalp_texture_mixed, 
        density_map_mixed, 
        difflocks.strand_codec, 
        normalization_dict=difflocks.normalization_dict, 
        scalp_mesh_data=difflocks.scalp_mesh_data, 
        tbn_space_to_world_func=tbn_space_to_world, 
        nr_chunks=difflocks.nr_chunks_decode_strands, 
        upsample_multiplier=3)
    
    # 步驟 7: 輸出最終結果
    npz_out_path=os.path.join(out_path, "mixed_output_strands.npz")
    np.savez(npz_out_path, positions=strand_points_world.cpu().numpy())
    print(f"混合後的 3D 髮束已儲存到: {npz_out_path}")


    # Create .blend
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
    
    # 建構命令
    cmd=[
        args_mock.blender_path, 
        "-t", str(args_mock.blender_nr_threads), 
        "--background", 
        "--python", "./inference/npz2blender.py",
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
        
    print("\n--- 4. 開始在背景執行 Blender 匯入 NPZ 檔案 ---")
    import subprocess
    subprocess.run(cmd, capture_output=False)
    print("Blender import script executed successfully.")


#=====================================================================================
#======================== Revision Region End ========================================
#=====================================================================================


if __name__ == '__main__':

    # run()
    run_kung()