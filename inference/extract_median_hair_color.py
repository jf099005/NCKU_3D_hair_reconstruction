#!/usr/bin/env python3

# 用 img2hair_kung_mul2.py 的中位數作法算出單張照片的髮色：
# mediapipe 偵測臉 -> crop_face(770) -> 髮型分割(類別 1 = 頭髮) -> 頭髮像素 RGB 取中位數。
# 結果寫到 <out_path>/hair_color_median.json (rgb 範圍 [0,1]) 與一張色塊圖，
# 給 render_multiview_blender.py --hair_rgb 使用。
#
# Usage: python ./inference/extract_median_hair_color.py --img_path <img> --out_path <dir>

import argparse
import json
import os

import cv2
import numpy as np

from img2hair_kung_mul2 import Mediapipe, crop_face, VisionRunningMode


def extract_median_rgb(mediapipe_img, frame_rgb):
    _, face_landmarks = mediapipe_img.run(frame_rgb)
    if face_landmarks is None:
        print("WARNING: no face detected, using the full resized image")
    frame_cropped = crop_face(frame_rgb, face_landmarks, output_size=770)

    # 與 img2hair_kung_mul2.DiffLocksInference.extract_color_map 相同
    category_mask = mediapipe_img.run_segmentation(frame_cropped)
    hair_pixels = frame_cropped[category_mask == 1]
    if len(hair_pixels) > 0:
        extracted_rgb = np.median(hair_pixels, axis=0).astype(np.uint8)
    else:
        print("WARNING: no hair pixels found, falling back to dark gray")
        extracted_rgb = np.array([50, 50, 50], dtype=np.uint8)

    return extracted_rgb, frame_cropped, category_mask, len(hair_pixels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--out_path', required=True)
    args = parser.parse_args()

    frame = cv2.imread(args.img_path)
    if frame is None:
        raise FileNotFoundError(args.img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mediapipe_img = Mediapipe(VisionRunningMode.IMAGE)
    rgb_u8, frame_cropped, category_mask, nr_hair_pixels = extract_median_rgb(mediapipe_img, frame)

    os.makedirs(args.out_path, exist_ok=True)
    data = {
        "rgb": (rgb_u8.astype(np.float64) / 255.0).tolist(),
        "rgb_uint8": rgb_u8.tolist(),
        "nr_hair_pixels": int(nr_hair_pixels),
    }
    path_json = os.path.join(args.out_path, "hair_color_median.json")
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    swatch = np.zeros((256, 256, 3), dtype=np.uint8)
    swatch[:] = rgb_u8
    cv2.imwrite(os.path.join(args.out_path, "hair_color_median.png"), cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR))
    hair_mask_vis = (category_mask == 1).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(args.out_path, "hair_mask_median.png"), hair_mask_vis)

    print(f"median hair color: {rgb_u8.tolist()} (from {nr_hair_pixels} hair pixels) -> {path_json}")


if __name__ == '__main__':
    main()
