# ===================================================================
# preprocess_pages.py
#
# 核心功能 (Core Features):
#   - 自動將 PDF (補習班生字練習簿等) 轉換為逐頁 PNG。
#   - 方向校正：
#       * 多重檢查 (標籤列 OCR → 全圖 OCR → 空白率 → Tesseract OSD)。
#       * 自動判斷是否需要 180° 旋轉，避免倒置。
#   - Debug 輔助：會輸出帶有旋轉資訊的圖片到 debug_steps/。
#
# 運行流程 (Execution Flow):
#   1. 初始化環境：
#        - 確認輸入 PDF 是否存在。
#        - 建立輸出資料夾 data/... 和 debug_steps/。
#        - 若已存在舊檔案則清空。
#
#   2. PDF 轉換：
#        - 使用 pdf2image 將 PDF 每頁轉為 PIL Image。
#        - 轉為 OpenCV 格式 (BGR) 以利處理。
#
#   3. 頁面校正：
#        - correct_orientation(): 多步驟串聯判斷，決定是否需要旋轉。
#        - correct_skew(): (已移除，保留介面)。
#
#   4. 儲存：
#        - 每頁輸出兩份：
#            * 校正後 PNG → data/cramschool_merged/
#            * 附註旋轉角度的 debug 圖片 → debug_steps/
#
#   5. 完成：
#        - 輸出處理狀態與完成訊息。
# ===================================================================
import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import shutil
import math

# --- 1. 環境設定與路徑定義 ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r"C:\poppler\poppler-25.07.0\Library\bin"

script_dir = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = os.path.join(script_dir, "pdf", "cramschool_merged.pdf")
PAGES_DIR = os.path.join(script_dir, "data", "cramschool_merged")
DEBUG_DIR = os.path.join(script_dir, "debug_steps")

# ========================= 專家加權投票校正系統 (修正版) =========================

def correct_orientation(image):
    """保守版 v5: 串聯判斷，順序=標籤列→OCR→空白率→OSD"""
    print("    -> 執行保守版 v5 串聯判斷進行方向校正...")

    rotated_180_image = cv2.flip(image, -1)
    h, w = image.shape[:2]
    small_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small_rotated_180 = cv2.flip(small_image, -1)

    def avg_confidence(img):
        data = pytesseract.image_to_data(
            img, lang='chi_tra', config='--psm 6',
            output_type=pytesseract.Output.DICT
        )
        confs = [float(c) for c in data['conf'] if c != '-1']
        return np.mean(confs) if confs else 0.0

    try:
        # --- Step 1: 標籤列 vs 底部列 ---
        row_height = h // 15  # 取大約 1/15 高度
        top_row = image[:row_height, :]
        bottom_row = image[-row_height:, :]
        top_conf = avg_confidence(top_row)
        bottom_conf = avg_confidence(bottom_row)
        print(f"      [標籤列檢查] 上行={top_conf:.2f}, 下行={bottom_conf:.2f}")

        if top_conf >= bottom_conf:
            print("        -> 標籤列較清楚，保持原樣。")
            return image, 0
        print("        -> 底部更清楚，進入下一步驗證...")

        # --- Step 2: OCR 信心度驗證 ---
        conf_normal = avg_confidence(small_image)
        conf_rotated = avg_confidence(small_rotated_180)
        print(f"      [OCR] 正常={conf_normal:.2f}, 旋轉後={conf_rotated:.2f}")

        if conf_rotated <= conf_normal:
            print("        -> OCR 驗證未提升，保持原樣。")
            return image, 0
        print("        -> OCR 驗證通過，進入空白率檢查...")

        # --- Step 3: 空白率檢查 (上下 1/4) ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        top_quarter = binary[:h//4, :]
        bottom_quarter = binary[3*h//4:, :]
        top_blank_ratio = cv2.countNonZero(top_quarter) / top_quarter.size
        bottom_blank_ratio = cv2.countNonZero(bottom_quarter) / bottom_quarter.size
        print(f"      [空白率] 上={top_blank_ratio:.2f}, 下={bottom_blank_ratio:.2f}")

        if not (bottom_blank_ratio < top_blank_ratio):
            print("        -> 空白率檢查不支持旋轉，保持原樣。")
            return image, 0
        print("        -> 空白率檢查通過，進入 OSD 最終確認...")

        # --- Step 4: OSD 最終確認 ---
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = osd.get('rotate', 0)
        confidence = osd.get('confidence', 0.0)
        print(f"      [OSD] rotation={rotation}, confidence={confidence:.2f}")

        if confidence < 40.0:
            print("        -> OSD 信心不足，保持原樣。")
            return image, 0
        if rotation == 180:
            print("        -> OSD 確認為 180°，執行旋轉。")
            return rotated_180_image, 180
        else:
            print("        -> OSD 不支持旋轉，保持原樣。")
            return image, 0

    except Exception as e:
        print(f"    -> 校正發生錯誤: {e}, 將回傳原始圖片。")
        return image, 0




def correct_skew(image):
    """(功能已移除)"""
    print("    -> 傾斜校正已移除，跳過此步驟。")
    return image

# --- 主執行流程 ---
def run_preprocessing():
    # ... (主流程不變) ...
    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print("✔ 環境與資料夾設定完成。")
    if os.listdir(PAGES_DIR) or os.listdir(DEBUG_DIR):
        print(f"清空舊的 '{os.path.basename(PAGES_DIR)}' 和 '{os.path.basename(DEBUG_DIR)}' 資料夾...")
        if os.path.isdir(PAGES_DIR): shutil.rmtree(PAGES_DIR)
        if os.path.isdir(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(PAGES_DIR, exist_ok=True)
        os.makedirs(DEBUG_DIR, exist_ok=True)
    try:
        pages_in_memory = convert_from_path(pdf_path=PDF_FILE, dpi=300, poppler_path=POPPLER_PATH)
        print(f"  -> 成功讀取 {len(pages_in_memory)} 頁。")
        print("  -> 開始逐頁校正並儲存為 PNG...")
        for i, page_image_pil in enumerate(pages_in_memory):
            page_num = i + 1
            print(f"\n  -- 處理第 {page_num} 頁 --")
            image_raw = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
            image_oriented, rotation_angle = correct_orientation(image_raw)
            image_final = correct_skew(image_oriented)
            annotated = image_final.copy()
            cv2.putText(annotated, f"Rotation: {rotation_angle} deg", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10, cv2.LINE_AA)
            cv2.imwrite(os.path.join(DEBUG_DIR, f'page_{page_num:03d}_annotated.png'), annotated)
            final_output_path = os.path.join(PAGES_DIR, f'page_{page_num:03d}.png')
            is_success, buffer = cv2.imencode('.png', image_final)
            if is_success:
                with open(final_output_path, 'wb') as f:
                    f.write(buffer)
            print(f"    -> 已儲存校正後的圖片: {os.path.basename(final_output_path)}")
    except FileNotFoundError:
        print(f"  -> ❌ 錯誤：找不到 PDF 檔案 '{PDF_FILE}'。")
    except Exception as e:
        print(f"  -> ❌ 錯誤：PDF 預處理失敗: {e}")
    print("\n✔ 所有頁面預處理完成！")

if __name__ == '__main__':
    run_preprocessing()