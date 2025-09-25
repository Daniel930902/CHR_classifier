# ===================================================================
# debug_grid.py
#
# 核心功能 (Core Features):
#   - 從指定資料夾中隨機挑選一張掃描圖片 (或手動固定 page_001.png)。
#   - 對圖片進行前處理：
#       * 灰階化
#       * 自適應二值化 (adaptiveThreshold)
#       * 腐蝕 (erode) → 細化線條
#       * 閉運算 (morphologyEx with CLOSE) → 修補格線斷裂
#   - 偵測所有輪廓 (findContours)，並依照面積與長寬比篩選可能的格子。
#   - 在原始圖上以綠色框標示出符合條件的格子。
#   - 顯示最終結果，方便調整核心參數 (面積範圍、aspect ratio)。
#
# 運行流程 (Execution Flow):
#   1. 從 data/cramschool_merged/ 讀取圖片檔案。
#   2. 隨機挑選一張圖片 (程式中目前固定 page_001.png)。
#   3. 對圖片進行前處理與二值化。
#   4. 偵測所有輪廓，篩選可能的格子。
#   5. 在圖片上標註格子，並以 OpenCV 視窗顯示。
#   6. 使用者可依據效果調整 if 條件 (面積、aspect ratio)，直到正確框選所有格子。
# ===================================================================

import cv2
import os
import random

# --- 設定 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(script_dir, "data", "cramschool_merged") # 設定圖片所在的資料夾名稱
DISPLAY_WIDTH = 500 

# ========================= 從資料夾中隨機選擇一張圖片 =========================
try:
    # 取得資料夾中所有檔案，並篩選出圖片檔
    all_files = os.listdir(IMAGE_DIR)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"錯誤：在 '{IMAGE_DIR}' 資料夾中找不到任何支援的圖片檔案。")
        exit() # 如果沒有圖片，就結束程式

    # 3. 從圖片列表中隨機挑選一個
    random_image_name = random.choice(image_files)
    random_image_name = "page_001.png"
    
    # 4. 組合出完整的圖片路徑
    IMAGE_TO_DEBUG = os.path.join(IMAGE_DIR, random_image_name)
    
    print(f"--- 本次隨機選取的圖片是: {IMAGE_TO_DEBUG} ---")

except FileNotFoundError:
    print(f"錯誤：找不到 '{IMAGE_DIR}' 資料夾。請確認程式旁邊有名為 'pages' 的資料夾。")
    exit()
# =========================================================================


# --- 用於縮放圖片以便顯示的函式 ---
def resize_for_display(image, width):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w, _ = image.shape
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

# 1. 讀取圖片
image = cv2.imread(IMAGE_TO_DEBUG)
if image is None:
    print(f"錯誤：無法讀取圖片 {IMAGE_TO_DEBUG}")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 進行二值化處理 
    thresh = cv2.adaptiveThreshold( gray, 300 , cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15 , 10 )

    # 建立一個較小的核心用於腐蝕，目的是讓所有線條先變細
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ( 3, 2 ))
    # 執行腐蝕， iterations=1 表示腐蝕一次
    eroded_thresh = cv2.erode(thresh, erode_kernel, iterations = 1 )

    # (原步驟) 建立用於閉運算的核心
    close_kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 9, 7 ) )
    # (原步驟) 執行閉運算，但這次的對象是已經被腐蝕變細的圖像
    fixed_thresh = cv2.morphologyEx(eroded_thresh, cv2.MORPH_CLOSE, close_kernel, iterations = 2 )

    # (可選) 顯示標準化後的二值圖，方便除錯
    # cv2.imshow('Standardized Threshold', resize_for_display(fixed_thresh.copy(), DISPLAY_WIDTH))

    # 3. 尋找所有輪廓 (使用處理後的 fixed_thresh)
    contours, _ = cv2.findContours( fixed_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
        
    # print(f"\n找到了 {len(contours)} 個輪廓。")
    
    # 4. 遍歷所有輪廓
    debug_image = image.copy()
    count_valid = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # --- 關鍵修改處 ---
        if 45000 < area < 65000 and 0.95 < aspect_ratio < 1.05 :
            # print(f"Found box {count_valid} Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 4 ) 
            count_valid += 1
        # ---
            
    print(f"\n標示出 {count_valid} 個符合目前篩選條件的輪廓 (綠色)。")
    
    # 5. 顯示最終結果
    debug_image_resized = resize_for_display(debug_image, DISPLAY_WIDTH)
    cv2.imshow('Filtered Contours (Resized)', debug_image_resized)
    print("\n顯示篩選後的輪廓... 按任意鍵結束程式。")
    # print("你的目標是調整程式碼中的 if 條件，讓所有的小方格都出現綠色框。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()