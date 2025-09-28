# ===================================================================
#   debug_grid.py (Matplotlib 版本顯示)
# ===================================================================
import cv2
import os
import random
import matplotlib.pyplot as plt  # ✅ 改用 Matplotlib

# --- 設定 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(script_dir, "data", "250928")  # 設定圖片所在的資料夾名稱
DISPLAY_WIDTH = 500

# ========================= 從資料夾中隨機選擇一張圖片 =========================
try:
    # 取得資料夾中所有檔案，並篩選出圖片檔
    all_files = os.listdir(IMAGE_DIR)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"錯誤：在 '{IMAGE_DIR}' 資料夾中找不到任何支援的圖片檔案。")
        exit()  # 如果沒有圖片，就結束程式

    # 3. 從圖片列表中隨機挑選一個
    random_image_name = random.choice(image_files)
    random_image_name = "001.png"

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
    SCALE_FACTOR = 1.5
    image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 進行二值化處理
    thresh = cv2.adaptiveThreshold(gray, 300, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 23, 15 )

    # 建立一個較小的核心用於腐蝕，目的是讓所有線條先變細
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    # 執行腐蝕， iterations=1 表示腐蝕一次
    eroded_thresh = cv2.erode(thresh, erode_kernel, iterations=1)

    # (原步驟) 建立用於閉運算的核心
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 5))
    # (原步驟) 執行閉運算，但這次的對象是已經被腐蝕變細的圖像
    fixed_thresh = cv2.morphologyEx(eroded_thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 3. 尋找所有輪廓 (使用處理後的 fixed_thresh)
    contours, _ = cv2.findContours(fixed_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 遍歷所有輪廓
    debug_image = image.copy()
    count_valid = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        # --- 關鍵修改處 ---
        if 20000 < area < 90000 and 0.90 < aspect_ratio < 1.1:
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 4)
            count_valid += 1
        # ---

    print(f"\n標示出 {count_valid} 個符合目前篩選條件的輪廓 (綠色)。")

    # 5. 顯示最終結果 (用 Matplotlib)
    debug_image_resized = resize_for_display(debug_image, DISPLAY_WIDTH)
    plt.imshow(cv2.cvtColor(debug_image_resized, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Contours (Resized)")
    plt.axis("off")
    plt.show()
