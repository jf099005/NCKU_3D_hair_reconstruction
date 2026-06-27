from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
import argparse
import os

# --- 顏色格式轉換 (與前一個版本相同) ---
def convert_rgb_to_hex(rgb_tuple):
    # ... (程式碼與前一個版本相同) ...
    r, g, b = [max(0, min(255, c)) for c in rgb_tuple[:3]] 
    return f"#{r:02X}{g:02X}{b:02X}"

# --- K-Means 顏色提取核心函式 (與前一個版本相同) ---
def extract_palette(image_path, n_colors, max_runs=10):
    # ... (程式碼與前一個版本相同) ...
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"錯誤：找不到檔案路徑 '{image_path}'")

    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels = np.array(list(img.getdata())) 

    kmeans = KMeans(n_clusters=n_colors, 
                    random_state=42, 
                    max_iter=max_runs,
                    n_init='auto')
    
    kmeans.fit(pixels)
    
    # 返回 RGB 列表，以便後續繪圖使用
    main_colors_rgb = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
    
    return main_colors_rgb


# --- ✨ 新增功能：建立結果圖片 ---
def create_palette_image(original_img_path, rgb_palette, output_path):
    """
    建立一張新的圖片，將顏色色板繪製在原始圖片的上方。
    
    Args:
        original_img_path (str): 原始圖片的路徑。
        rgb_palette (list): 包含 (R, G, B) 元組的顏色列表。
        output_path (str): 輸出結果圖片的路徑。
    """
    try:
        # 1. 讀取原始圖片
        original_img = Image.open(original_img_path)
        img_width, img_height = original_img.size
        
        # 2. 定義色板區域的高度和間距
        palette_height = 80 # 色板區域的高度
        padding = 10        # 邊緣和方塊之間的間距
        
        # 3. 建立新的圖片 (原始圖片 + 色板區域)
        new_height = img_height + palette_height + padding
        # 建立白色背景的圖片
        output_img = Image.new('RGB', (img_width, new_height), 'white')
        
        # 4. 繪製原始圖片
        # 將原始圖片貼到新圖片的下方 (從 palette_height + padding 開始)
        output_img.paste(original_img, (0, palette_height + padding))
        
        # 5. 繪製顏色方塊
        draw = ImageDraw.Draw(output_img)
        num_colors = len(rgb_palette)
        
        # 計算每個顏色方塊的寬度和位置
        bar_width = (img_width - 2 * padding) // num_colors
        
        for i, color_rgb in enumerate(rgb_palette):
            # 定義方塊的邊界
            x0 = padding + i * bar_width
            y0 = padding
            x1 = padding + (i + 1) * bar_width - padding // 2 # 每個方塊之間留點空隙
            y1 = palette_height
            
            # 繪製方塊
            draw.rectangle([(x0, y0), (x1, y1)], fill=color_rgb)
            
        # 6. 儲存結果圖片
        output_img.save(output_path)
        return True

    except Exception as e:
        print(f"❌ 繪製結果圖片時發生錯誤: {e}")
        return False


# --- 主執行區塊與參數解析 (進行修改) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 K-Means 演算法從圖像中提取主要的顏色色板，並輸出視覺化結果。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 定義參數 (與前一個版本相同)
    parser.add_argument('image_path', type=str, help="輸入圖片的檔案路徑")
    parser.add_argument('-n', '--num-colors', type=int, default=5, help="欲提取的主要顏色數量 (預設為 5)。")
    parser.add_argument('-r', '--max-runs', type=int, default=10, help="K-Means 演算法的最大迭代次數 (maxRuns)。")
    
    # ✨ 新增輸出路徑參數
    parser.add_argument('-o', '--output-path', type=str, default="output_palette.png", 
                        help="結果圖片的輸出路徑和檔名 (預設為 output_palette.png)。")

    args = parser.parse_args()
    
    print(f"--- 🎨 顏色提取開始 ---")
    # ... (輸出略) ...

    try:
        # 1. 呼叫核心提取函式，取得 RGB 格式的色板
        rgb_palette = extract_palette(
            args.image_path, 
            args.num_colors, 
            args.max_runs
        )
        
        # 2. 轉換為 Hex 格式並印出 (純文字結果)
        hex_palette = [convert_rgb_to_hex(rgb) for rgb in rgb_palette]
        print("\n✅ 提取結果 (Hex 色板):")
        for i, hex_color in enumerate(hex_palette):
            print(f"   [{i+1}] {hex_color}")
        
        # 3. 呼叫繪圖函式，建立視覺化結果圖片
        success = create_palette_image(
            args.image_path, 
            rgb_palette, 
            args.output_path
        )

        if success:
            print(f"\n🎉 視覺化結果已成功儲存至: {args.output_path}")
        else:
            print("\n⚠️ 視覺化圖片輸出失敗。")

            
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤: {e}")
    except Exception as e:
        print(f"\n❌ 處理圖片時發生錯誤: {e}")