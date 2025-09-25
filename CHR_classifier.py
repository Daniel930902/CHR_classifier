# ===================================================================
# CHR_class.py
#
# 核心功能 (Core Features):
#    - 自動化批次處理手寫字練習簿的掃描圖片。
#    - 三通道格子偵測 (輪廓分析法 / Hough 直線變換 / 投影剖面法)，
#      自動切割出標籤列與練習格。
#    - OCR 辨識標籤格，並支援白名單 (whitelist) 與智慧推斷：
#        * 使用者可指定第一個字作為全域錨點。
#        * 提供模式記憶與錨點校正，提升序列推斷準確度。
#    - 嚴格的空白檢測 (多特徵：筆畫持久度 / 邊緣密度 / 連通元件)，
#      避免將空格或殘影誤判為字跡。
#    - 自動裁切與分類儲存字跡圖片，並建立以字元為單位的資料夾。
#    - 產出詳細統計報告：
#        * 儲存率 (已存 / 實際有字跡)
#        * 資料產出率 (已存 / 理論格數 - 空白 - 未知標籤)
#        * 低存量欄位報告 (不足 10 個字跡的字元會特別標註)
#    - Debug 輔助：輸出帶有標註框的標籤偵測圖。
#
# 運行流程 (Execution Flow):
#    1. 初始化環境：
#         - 建立輸出資料夾 (datasets/...) 與 debug 資料夾。
#         - 載入 whitelist.txt，詢問是否啟用白名單模式。
#         - 使用者可選擇輸入第一個字元作為全域錨點。
#
#    2. 預處理檢查：
#         - 若未找到校正後的 PNG，會自動呼叫 preprocess_pages.py。
#
#    3. 格子偵測：
#         - 依序嘗試三種方法找出 9×10 的格子矩陣。
#
#    4. 標籤辨識與推斷：
#         - Tesseract OCR 嘗試辨識每個標籤格。
#         - 啟用白名單模式時，會透過序列比對 / 錨點校正 / 模式記憶
#           來推斷最終標籤。
#         - 空白或信心度不足的標籤會標記為 '?'。
#
#    5. 練習格處理：
#         - 若標籤是 '?' → 整欄練習格跳過。
#         - 否則逐格判斷是否為空白，保留有效字跡並裁切儲存。
#
#    6. 統計與報告：
#         - 總頁數、總格子數、標籤數、練習格數。
#         - 成功辨識的標籤欄位數 vs 未知標籤欄位數。
#         - 可定址格子數 vs 空白 vs 已儲存數。
#         - 儲存率、資料產出率。
#         - 低存量欄位警告。
#
# ===================================================================


import os
import cv2
import pytesseract
import subprocess
import sys
import numpy as np
import shutil
import unicodedata
import math

# --- 1. 環境設定 ---

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
script_dir = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(script_dir, "data", "cramschool_merged")
OUTPUT_DIR = os.path.join( "E:\datasets", "cramschool_merged")
DEBUG_DIR = os.path.join(script_dir, "debug_steps")
WHITELIST_FILE = os.path.join(script_dir, "whitelist.txt")

# --- 2. 讀取並載入白名單 ---

use_whitelist = False
whitelist_text = ""
global_offset = None # 用於循序推斷的全域偏移量
try:
    with open(WHITELIST_FILE, 'r', encoding='utf-8') as f:
        whitelist_text = "".join(f.read().split())
    print(f"✔ 成功載入 {len(whitelist_text)} 個白名單字元。")

    choice = input("是否啟用白名單推斷功能？(直接按 Enter 表示是, 輸入 n 表示否): ").strip().lower()
    if choice == 'n':
        use_whitelist = False
        print(" -> 白名單功能已停用。")
    else:
        use_whitelist = True
        print(" -> 白名單推斷功能已啟用。")
        
        first_char = input("請輸入該批資料集的【第一個字】(可留空，直接按 Enter 則每頁獨立尋找錨點): ").strip()
        if use_whitelist and first_char:
            # 對使用者輸入的字元也進行標準化，以確保能正確匹配
            normalized_first_char = unicodedata.normalize("NFKC", first_char)
            if normalized_first_char in whitelist_text:
                global_offset = whitelist_text.find(normalized_first_char)
                print(f"✔ 已設定全域起始錨點為 '{normalized_first_char}' (索引: {global_offset})，將進行循序推斷。")
            else:
                print(f"❌ 警告: 起始字 '{first_char}' 不在白名單中，將退回至每頁獨立尋找錨點模式。")

except FileNotFoundError:
    print(f"❌ 警告：找不到白名單檔案 '{WHITELIST_FILE}'，無法使用白名單功能。")
    use_whitelist = False

# --- 3. 建立並清空必要資料夾 ---

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
print("✔ 環境與資料夾設定完成。")

print(f"清空舊的 '{os.path.basename(OUTPUT_DIR)}' 和 '{os.path.basename(DEBUG_DIR)}' 資料夾...")
if os.path.isdir(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
if os.path.isdir(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ======================== 核心工具函式 =========================

def prepare_roi_for_ocr(full_img, box):
    """為 OCR 準備高品質的 ROI"""
    x, y, w, h = box
    roi = full_img[y:y+h, x:x+w]
    m = int(min(h, w) * 0.12)
    if m > 0 and h > 2*m and w > 2*m:
        roi = roi[m:h-m, m:w-m]

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(b) < 127: b = cv2.bitwise_not(b)
    b = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value = 255 )
    return b

def ocr_char_and_conf(img_bin):
    """對單一 ROI 進行 OCR，回傳(字元, 信心度)"""
    cfg = "--oem 1 --psm 10"
    data = pytesseract.image_to_data(img_bin, lang='chi_tra', config=cfg, output_type=pytesseract.Output.DICT)

    confs = [int(c) for i, c in enumerate(data['conf']) if int(c) > -1 and data['text'][i].strip()]
    text = "".join(t for t in data['text'] if t.strip())
    char = "".join(c for c in text if '\u4e00' <= c <= '\u9fff')

    final_char = char[0] if char else ""
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return final_char, mean_conf

def _persistence_mask(gray, ksizes=(25, 41), min_keep=2):
    """
    多重二值化的 '持久度' 估計：把多種門檻的結果做 AND/OR，比較穩定的墨水才算數。
    回傳 (persistence_ratio, union_ratio)
    """
    # 三種二值化：Otsu + 兩種 adaptive mean (不同 window)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ams = []
    for k in ksizes:
        k = max(15, k | 1)  # 奇數
        am = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, k, 10)
        ams.append(am)
    stack = [otsu] + ams
    union = np.zeros_like(otsu)
    votes = np.zeros_like(otsu, dtype=np.uint8)
    for m in stack:
        union = cv2.bitwise_or(union, m)
        votes = cv2.add(votes, (m > 0).astype(np.uint8))
    keep = (votes >= min_keep).astype(np.uint8) * 255
    inter = cv2.bitwise_and(union, keep)

    inter_cnt = int(cv2.countNonZero(inter))
    union_cnt = int(cv2.countNonZero(union))
    persistence_ratio = (inter_cnt / union_cnt) if union_cnt > 0 else 0.0
    return persistence_ratio, (union_cnt / (gray.shape[0] * gray.shape[1]) if gray.size else 0.0)


def _stroke_stats(gray):
    """
    邊緣 + 連通元件的筆畫統計。回傳 (edge_density, n_cc, max_cc_area_ratio)
    """
    # 邊緣
    edges = cv2.Canny(gray, 60, 180)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # 連通元件（在比較乾淨的二值圖上）
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 消除鹽胡椒
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # 去掉背景，並忽略微小雜訊
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 20]
    n_cc = len(areas)
    max_cc = max(areas) if areas else 0
    max_cc_area_ratio = max_cc / float(gray.size)
    return edge_density, n_cc, max_cc_area_ratio


def is_label_blank_ultra_strict(gray,
                                std_thresh=28,
                                union_ink_ratio_min=0.040,
                                persistence_min=0.70,
                                edge_density_min=0.012,
                                n_cc_min=1,
                                max_cc_area_ratio_min=0.005):
    if gray is None or gray.size == 0:
        return True

    if np.std(gray) < std_thresh:
        return True

    persis, union_ratio = _persistence_mask(gray)
    if union_ratio < union_ink_ratio_min:
        return True
    if persis < persistence_min:
        return True

    edge_density, n_cc, max_cc_area_ratio = _stroke_stats(gray)
    if edge_density < edge_density_min:
        return True
    if n_cc < n_cc_min:
        return True
    if max_cc_area_ratio < max_cc_area_ratio_min:
        return True

    return False


def is_grid_blank_dynamically(gray,
                              std_thresh=25,
                              union_ink_ratio_min=0.020,
                              persistence_min=0.60,
                              edge_density_min=0.010,
                              n_cc_min=1,
                              max_cc_area_ratio_min=0.004):
    """
    (v16.7.0) 基於多重特徵的動態空白檢查，專為手寫字跡設計。
    此機制與標籤列的 is_label_blank_ultra_strict 邏輯相同，但閥值針對手寫特性調整。
    核心是透過筆畫持久度、邊緣密度、連通元件等多維度特徵來判斷「筆畫量」，
    而非單純的墨水比例，能更準確地分辨微弱/潦草字跡與純粹的雜訊/空白。
    """
    if gray is None or gray.size == 0:
        return True

    if np.std(gray) < std_thresh:
        return True

    persis, union_ratio = _persistence_mask(gray)
    if union_ratio < union_ink_ratio_min:
        return True
    if persis < persistence_min:
        return True

    edge_density, n_cc, max_cc_area_ratio = _stroke_stats(gray)
    if edge_density < edge_density_min:
        return True
    if n_cc < n_cc_min:
        return True
    if max_cc_area_ratio < max_cc_area_ratio_min:
        return True

    return False

def find_grid_boxes_by_contours(image, params):
    """通道一：輪廓分析法"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.medianBlur( gray, 3 )

    thresh = cv2.adaptiveThreshold( denoised_gray, 
                                      255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 
                                      27, 23 )
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ( 5, 1 ))
    eroded_thresh = cv2.erode( thresh, erode_kernel, iterations = 1 )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ( 8, 5 ))
    fixed_thresh = cv2.morphologyEx(eroded_thresh, cv2.MORPH_CLOSE, close_kernel, iterations = 2 )
    contours, _ = cv2.findContours(fixed_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    grid_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area, aspect_ratio = w * h, w / h if h > 0 else 0
        if params['min_area'] < area < params['max_area'] and params['min_ratio'] < aspect_ratio < params['max_ratio']:
            grid_boxes.append((x, y, w, h))
    return grid_boxes

def find_grid_boxes_by_hough(image, params):
    """通道二：霍夫直線變換法 (Fallback)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.medianBlur(gray, 3)
    edges = cv2.Canny( denoised_gray, 50, 100, apertureSize = 3 )
    lines = cv2.HoughLinesP( edges, 1, np.pi / 180, 
                             threshold = 110, 
                             minLineLength = int( image.shape[0]//4 ),
                             maxLineGap = 50 )

    if lines is None: return []

    def cluster_lines(lines, threshold=params['cluster_thresh']):
        if not lines: return []
        lines = sorted(lines)
        clusters, current_cluster = [], [lines[0]]
        for line_pos in lines[1:]:
            if line_pos - current_cluster[-1] < threshold: current_cluster.append(line_pos)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [line_pos]
        clusters.append(int(np.mean(current_cluster)))
        return clusters

    horz_lines, vert_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) if x2 - x1 != 0 else 90
        if angle < 15 or angle > 165: horz_lines.append(y1)
        elif 75 < angle < 105: vert_lines.append(x1)

    h_lines = cluster_lines(horz_lines)
    v_lines = cluster_lines(vert_lines)

    hough_boxes = []
    if len(h_lines) > 1 and len(v_lines) > 1:
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                y1, y2 = h_lines[i], h_lines[i+1]
                x1, x2 = v_lines[j], v_lines[j+1]
                w, h = x2 - x1, y2 - y1
                if w > 50 and h > 50: hough_boxes.append((x1, y1, w, h))
    return hough_boxes

def find_grid_boxes_by_projection(image, params):
    """通道三：投影剖面法 (最終王牌)"""
    print("  -> [通道 3] 執行投影剖面法...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    horz_proj, vert_proj = np.sum(binary, axis=1), np.sum(binary, axis=0)

    def find_peaks(projection, min_dist, threshold_ratio=0.3):
        threshold = np.max(projection) * threshold_ratio
        peaks = []
        for i in range(1, len(projection) - 1):
            if projection[i] > threshold and projection[i] > projection[i-1] and projection[i] > projection[i+1]:
                if all(abs(i - p) >= min_dist for p in peaks):
                    peaks.append(i)
        return peaks

    avg_grid_side = (params['min_area']**0.5 + params['max_area']**0.5) / 2

    y_coords = find_peaks(horz_proj, int(avg_grid_side * 0.8))
    x_coords = find_peaks(vert_proj, int(avg_grid_side * 0.8))

    proj_boxes = []
    if len(y_coords) > 1 and len(x_coords) > 1:
        for i in range(len(y_coords) - 1):
            for j in range(len(x_coords) - 1):
                y1, y2, x1, x2 = y_coords[i], y_coords[i+1], x_coords[j], x_coords[j+1]
                proj_boxes.append((x1, y1, x2-x1, y2-y1))
    return proj_boxes

def find_grid_boxes(image):
    """三通道格子偵測系統"""
    params = {'min_area': 45000, 'max_area': 65000, 'min_ratio': 0.85, 'max_ratio': 1.10, 'cluster_thresh': 50 }

    # 通道一
    print("  -> [通道 1] 執行輪廓分析法...")
    grid_boxes = find_grid_boxes_by_contours(image, params)
    if len(grid_boxes) >= 99:
        print(f"  -> [通道 1] 成功找到 {len(grid_boxes)} 個格子。")
        return grid_boxes

    # 通道二
    print(f"  -> [通道 1] 失敗 (只找到 {len(grid_boxes)} 個格子)，切換至 [通道 2] Hough 直線變換法...")
    grid_boxes_hough = find_grid_boxes_by_hough(image, params)
    if len(grid_boxes_hough) > len(grid_boxes):
        print(f"  -> [通道 2] 找到了更多格子 ({len(grid_boxes_hough)})，採用其結果。")
        grid_boxes = grid_boxes_hough
    if len(grid_boxes) >= 99: return grid_boxes

    # 通道三
    print(f"  -> [通道 2] 失敗 (只找到 {len(grid_boxes)} 個格子)，切換至 [通道 3] 投影剖面法...")
    grid_boxes_proj = find_grid_boxes_by_projection(image, params)
    if len(grid_boxes_proj) > len(grid_boxes):
        print(f"  -> [通道 3] 找到了更多格子 ({len(grid_boxes_proj)})，採用其結果。")
        grid_boxes = grid_boxes_proj
        
    return grid_boxes

# ========================= 自動化預處理檢查 =========================

if not os.path.isdir(PAGES_DIR) or not os.listdir(PAGES_DIR):
    print(f"\nℹ️ 提示：找不到或 '{os.path.basename(PAGES_DIR)}' 資料夾是空的。")
    print(f"將自動執行 'preprocess_pages.py'...")
    preprocess_script_path = os.path.join(script_dir, 'preprocess_pages.py')
    if not os.path.isfile(preprocess_script_path):
        print(f"❌ 錯誤：找不到 {preprocess_script_path}"); exit()
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, preprocess_script_path],
            check=True, capture_output=True, text=True, encoding='utf-8', env=env
        )
        print("\n--- preprocess_pages.py 日誌 ---\n", result.stdout, "\n--- 日誌結束 ---")
    except subprocess.CalledProcessError as e:
        print("\n❌ 錯誤：執行預處理失敗\n", e.stderr); exit()
    print("\n✔ 預處理完成，繼續執行分析...")
else:
    print(f"\n✔ 偵測到 '{os.path.basename(PAGES_DIR)}' 已存在，直接進入分析。")

# ========================= 主處理迴圈 =========================

print("\n⏳ 開始分析已校正的圖片並切割字跡...")
char_counters = {}
page_files = sorted([f for f in os.listdir(PAGES_DIR) if f.endswith('.png')])

# 全局統計變數
total_pages_processed = 0
total_grids_found = 0
total_label_boxes_found = 0
total_practice_grids_found = 0
total_labels_recognized = 0
total_handwriting_saved = 0
total_blanks_skipped = 0
total_addressable_grids = 0
incomplete_columns_log = []
GRIDS_PER_PAGE_THEORY = 9 * 10

# <<< 新增：模式資料庫 >>>
pattern_database = []
last_known_anchor_index = -1

for page_idx, page_filename in enumerate(page_files):
    print(f"\n--- 正在分析頁面: {page_filename} ({page_idx + 1}/{len(page_files)}) ---")
    image = cv2.imread(os.path.join(PAGES_DIR, page_filename))
    if image is None:
        print(f"  -> 無法讀取圖片 {page_filename}，跳過。")
        continue
    total_pages_processed += 1

    # === 步驟 1: 三通道格子偵測 ===
    grid_boxes = find_grid_boxes(image)
    if len(grid_boxes) < 9:
        print(f"  -> 警告: 找到的格子數 ({len(grid_boxes)}) 過少，跳過此頁。"); continue
    total_grids_found += len(grid_boxes)
    grid_boxes.sort(key=lambda b: (b[1], b[0]))

    # === 步驟 2: 標籤列定位 (根據固定規則) ===
    COL_COUNT = 9

    # 分離標籤列與練習格
    first_row_boxes = grid_boxes[:COL_COUNT]
    practice_boxes = grid_boxes[COL_COUNT:]

    first_row_boxes.sort(key=lambda b: b[0])
        
    print(f"  -> 已確定標籤列: {len(first_row_boxes)} 個, 練習格: {len(practice_boxes)} 個")

    total_label_boxes_found += len(first_row_boxes)
    total_practice_grids_found += len(practice_boxes)

    # === 步驟 3: 標籤 OCR 與模式推斷 ===
    final_labels = ["?"] * len(first_row_boxes)
    is_determined = False

    # --- 初步 OCR ---
    ocr_results = []
    for i, label_box in enumerate(first_row_boxes):
        roi_bin = prepare_roi_for_ocr(image, label_box)
        ch, conf = ocr_char_and_conf(roi_bin)

        if ch and conf < 45:
            ch = None

        x, y, w, h = label_box
        label_roi = image[y:y+h, x:x+w]
        gray_label_roi = cv2.cvtColor(label_roi, cv2.COLOR_BGR2GRAY)
        if is_label_blank_ultra_strict(gray_label_roi):
            ch = None

        ocr_results.append(ch)

    print(f"  -> 初步 OCR 結果: [{' '.join(c if c else '?' for c in ocr_results)}]")

    # --- 預先處理空白標籤格 ---
    blanks_in_label_row = 0
    for i, label_box in enumerate(first_row_boxes):
        x, y, w, h = label_box
        label_roi = image[y:y+h, x:x+w]
        gray_label_roi = cv2.cvtColor(label_roi, cv2.COLOR_BGR2GRAY)
        if is_label_blank_ultra_strict(gray_label_roi):
            ocr_results[i] = None
            blanks_in_label_row += 1
    if blanks_in_label_row > 0:
        print(f"  -> 偵測到標籤列有 {blanks_in_label_row} 個空白格，已在序列配對中標記為忽略。")


    # --- 推斷流程開始 ---
    if use_whitelist and whitelist_text:
        if page_idx == 0 and global_offset is not None:
            print(f"  -> [優先層 1] 使用者指定起始字元，強制設定偏移量為 {global_offset}")
            for i in range(len(first_row_boxes)):
                inferred_idx = global_offset + i
                if 0 <= inferred_idx < len(whitelist_text):
                    final_labels[i] = whitelist_text[inferred_idx]
            is_determined = True
            last_known_anchor_index = global_offset
        
        if not is_determined and pattern_database:
            print(f"  -> [優先層 2] 嘗試匹配 {len(pattern_database)} 個已記憶的序列...")
            for pattern in pattern_database:
                if len(pattern) != len(ocr_results): continue
                matches = sum(1 for i in range(len(ocr_results)) if ocr_results[i] and ocr_results[i] == pattern[i])
                if matches >= len(first_row_boxes) // 2:
                    final_labels = pattern
                    is_determined = True
                    try:
                        first_valid_char = next(c for c in final_labels if c != '?')
                        last_known_anchor_index = whitelist_text.find(first_valid_char)
                    except StopIteration:
                        pass
                    print(f"  -> ✔️ 模式匹配成功！套用已知序列: [{' '.join(final_labels)}]")
                    break
        
        if not is_determined:
            print(f"  -> [優先層 3] 模式匹配失敗，啟用智慧錨點推斷...")
            
            label_results = []
            for i, label_box in enumerate(first_row_boxes):
                ch = ocr_results[i]
                idx, conf = None, -1
                if ch:
                    temp_roi_bin = prepare_roi_for_ocr(image, label_box)
                    _, conf = ocr_char_and_conf(temp_roi_bin)
                    
                    normalized_ch = unicodedata.normalize("NFKC", ch)
                    if normalized_ch in whitelist_text:
                        idx = whitelist_text.find(normalized_ch)
                label_results.append({'pos': i, 'char': ch, 'conf': conf, 'idx': idx})

            anchors = [res for res in label_results if res['idx'] is not None and res['conf'] >= 40.0]
            final_offset = None

            # --- 順序 3a: 容錯序列配對 (優先) ---
            MIN_MATCH_COUNT = 3
            best_score = -1
            best_offset = None

            if len(ocr_results) > 0 and len(whitelist_text) >= len(ocr_results):
                for offset in range(len(whitelist_text) - len(ocr_results) + 1):
                    current_score = 0
                    for i in range(len(ocr_results)):
                        if ocr_results[i] and ocr_results[i] == whitelist_text[offset + i]:
                            current_score += 1
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_offset = offset

            if best_score >= MIN_MATCH_COUNT:
                final_offset = best_offset
                print(f"        -> [層級 3a] 成功：序列配對找到最佳吻合位置 (位移: {final_offset}, 匹配數: {best_score}/{len(ocr_results) - blanks_in_label_row})")
            else:
                print(f"        -> [層級 3a] 失敗：未找到足夠的匹配 (最高僅 {best_score} 個)，無法確定序列。")

            # --- 順序 3b: 高信心度單點錨定 (次之) ---
            if final_offset is None and anchors:
                best_anchor = max(anchors, key=lambda a: a['conf'])
                HIGH_CONF_THRESHOLD = 85.0
                if best_anchor['conf'] >= HIGH_CONF_THRESHOLD:
                    final_offset = best_anchor['idx'] - best_anchor['pos']
                    print(f"        -> [層級 3b] 成功：採用高信心度錨點 '{best_anchor['char']}' (信心度: {best_anchor['conf']:.0f}%)")
                else:
                    print(f"        -> [層級 3b] 失敗：無錨點信心度超過 {HIGH_CONF_THRESHOLD}%。")

            # --- 順序 3c: 中位數偏移量校正 (Fallback) ---
            if final_offset is None:
                if anchors:
                    offsets = [anchor['idx'] - anchor['pos'] for anchor in anchors]
                    final_offset = int(np.median(offsets))
                    print(f"        -> [層級 3c] 採用中位數偏移量進行校正: {final_offset}")
                else:
                    print(f"        -> [層級 3c] 失敗：找不到任何有效錨點。")
            
            # --- 套用最終計算出的偏移量 ---
            if final_offset is not None:
                temp_labels = ["?"] * len(first_row_boxes)
                for i in range(len(first_row_boxes)):
                    inferred_idx = i + final_offset
                    if 0 <= inferred_idx < len(whitelist_text):
                        temp_labels[i] = whitelist_text[inferred_idx]
                
                if any(label != "?" for label in temp_labels):
                    final_labels = temp_labels
                    if final_labels not in pattern_database:
                        pattern_database.append(final_labels)
                        print(f"  -> ✔️ 錨點推斷成功！學習到新模式: [{' '.join(final_labels)}]")
                    last_known_anchor_index = final_offset

    # 如果標籤是預先判定的空白，即使推斷出了結果，也強制其為 '?'
    for i in range(len(ocr_results)):
        if ocr_results[i] is None:
            final_labels[i] = '?'

    if not any(label != "?" for label in final_labels):
        original_ocr_results = [item if item is not None else '?' for item in ocr_results]
        print("  -> 未啟用白名單或所有推斷失敗，使用單字辨識結果。")
        final_labels = original_ocr_results

    total_labels_recognized += sum(1 for lab in final_labels if lab != "?")

    debug_img = image.copy()
    for i, (x, y, w, h) in enumerate(first_row_boxes):
        label_text = final_labels[i]
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(debug_img, label_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3 )
    cv2.imwrite(os.path.join(DEBUG_DIR, f'{page_filename}_labels.png'), debug_img)

    # === 步驟 3.5: 整欄一致性審核（防止把空白列硬塞成一字） ===
    COLUMN_MIN_RATIO = 0.3
    for i, char_label in enumerate(final_labels):
        if char_label == "?": continue
        lx = first_row_boxes[i][0]
        column_boxes = [b for b in practice_boxes if abs(b[0] - lx) < 50]

        nonblank_cnt = 0
        for (px, py, pw, ph) in column_boxes:
            roi = image[py:py+ph, px:px+pw]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if not is_grid_blank_dynamically(gray_roi):
                nonblank_cnt += 1

        ratio = nonblank_cnt / len(column_boxes) if column_boxes else 0
        if ratio < COLUMN_MIN_RATIO:
            final_labels[i] = "?"
            
    # === (v16.9.0) 步驟 3.6: 計算可定址的練習格總數 (用於精確統計) ===
    for i, char_label in enumerate(final_labels):
        if char_label != "?":
            lx = first_row_boxes[i][0]
            column_boxes_count = len([b for b in practice_boxes if abs(b[0] - lx) < 50])
            total_addressable_grids += column_boxes_count

    # === 步驟 4: 切割與儲存 ===
    # 核心邏輯：此迴圈僅處理 final_labels 中被成功標註的字元。
    # 若標籤格在前面步驟中被判定為空白 (final_labels[i] == "?")，
    # 則下方的 'continue' 指令會直接跳過該欄，達成「標籤是空白則整欄不存」的需求。
    for i, char_label in enumerate(final_labels):
        if char_label == "?": continue
        lx = first_row_boxes[i][0]
        print(f"                [處理 '{char_label}' 欄位]")
        os.makedirs(os.path.join(OUTPUT_DIR, char_label), exist_ok=True)
        if char_label not in char_counters:
            char_counters[char_label] = 0
        column_boxes = [b for b in practice_boxes if abs(b[0] - lx) < 50]
        saved_count = 0
        
        for grid_idx, (px, py, pw, ph) in enumerate(column_boxes):
            roi = image[py:py+ph, px:px+pw]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            if not is_grid_blank_dynamically(gray_roi):
                char_counters[char_label] += 1
                output_filename = f"{char_counters[char_label]:03d}.png"
                output_path = os.path.join(OUTPUT_DIR, char_label, output_filename)
                is_success, buffer = cv2.imencode('.png', roi)
                if is_success:
                    with open(output_path, 'wb') as f: f.write(buffer)
                    saved_count += 1
            else:
                total_blanks_skipped += 1
        
        if saved_count > 0:
            total_handwriting_saved += saved_count
            print(f"                              -> '{char_label}' 儲存 {saved_count} 個字跡")
            if saved_count < 10:
                incomplete_columns_log.append({'page': page_filename, 'char': char_label, 'count': saved_count})
                print(f"                              -> ❗ 警告: '{char_label}' 欄儲存數量不足 ({saved_count}/10)，已記錄。")

print("\n✔ 全部頁面處理完成！")

# --- 低存量欄位報告 ---
if incomplete_columns_log:
    print("\n" + "="*50)
    print("--- ⚠️ 低存量欄位報告 (儲存數量 < 10) ---")
    print("="*50)
    sorted_log = sorted(incomplete_columns_log, key=lambda x: x['page'])
    for log in sorted_log:
        print(f"  - 頁面: {log['page']:<15} | 字元: '{log['char']}' | 儲存數量: {log['count']}/10")

    total_missing_chars = sum(10 - log['count'] for log in incomplete_columns_log)
    print("-" * 20)
    print(f"  -> 總共缺少 {total_missing_chars} 個字跡。")
    print("="*50)
else:
    if total_pages_processed > 0:
        print("\n✔ 恭喜！所有成功處理的欄位均儲存了 10 個或以上的字跡。")

# --- 最終成果統計報告 ---
print("\n" + "="*50)
print("--- 最終成果統計報告 ---")
print("="*50)
if total_pages_processed > 0:
    print(f"總共處理頁面數量: {total_pages_processed} 頁")
    print(f"總共找到的格子數(含標籤+練習): {total_grids_found} 個")
    print(f"  - 標籤列格子數: {total_label_boxes_found} 個")
    print(f"  - 練習格總數 (物理偵測): {total_practice_grids_found} 個")
    print("-" * 20)
    
    unresolved_labels = total_label_boxes_found - total_labels_recognized
    print(f"成功決定的標籤欄位數: {total_labels_recognized} 欄")
    if unresolved_labels > 0:
        print(f"未能決定的標籤欄位數: {unresolved_labels} 欄")
    
    unaddressable_grids = total_practice_grids_found - total_addressable_grids
    print(f"可定址練習格總數 (有標籤): {total_addressable_grids} 格")
    if unaddressable_grids > 0:
         print(f"不可定址練習格總數 (標籤未知): {unaddressable_grids} 格")

    print("-" * 20)
    print(f"--- 在 {total_addressable_grids} 個「可定址」格子中的分析 ---")
    print(f"✅ **最終成功儲存的字跡總數**: {total_handwriting_saved} 張")
    print(f"   - 因「空白」而跳過的格子數: {total_blanks_skipped} 個")
    
    total_non_blank_addressable_grids = total_addressable_grids - total_blanks_skipped
    print(f"   - 實際有字跡的格子總數: {total_non_blank_addressable_grids} 個")
    
    print("-" * 20)
    
    # 指標一：儲存率 (衡量程式執行效率)
    if total_non_blank_addressable_grids > 0:
        storage_rate = (total_handwriting_saved / total_non_blank_addressable_grids) * 100
        print(f"**字跡儲存率 (已儲存 / 實際有字跡)**: {storage_rate:.2f}%")
    else:
        print("**字跡儲存率**: N/A (沒有找到任何有字跡的可定址格子)")
    
    # 指標二：資料產出率 (衡量相對於理論值的捕獲能力)
    total_grids_theory_practice = GRIDS_PER_PAGE_THEORY * total_pages_processed
    denominator_yield = total_grids_theory_practice - total_blanks_skipped - unaddressable_grids
    if denominator_yield > 0:
        yield_rate = (total_handwriting_saved / denominator_yield) * 100
        print(f"**資料產出率 (已儲存 / (理論總數 - 空白 - 未知標籤))**: {yield_rate:.2f}%")
    else:
        print("**資料產出率**: N/A (分母為零或負數)")

else:
    print("沒有處理任何頁面，無法產生報告。")
print("="*50)