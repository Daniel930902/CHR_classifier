# ===================================================================
# pdf2png.py
#
# 核心功能 (Core Features):
#   - 將指定的 PDF 檔案逐頁轉換為 PNG 影像。
#   - 可指定 Poppler 的執行路徑 (Windows 必須提供)。
#   - 支援多執行緒加速 (thread_count 參數)。
#
# 運行流程 (Execution Flow):
#   1. 設定：
#        - Poppler 安裝路徑 (poppler_path)。
#        - 輸入 PDF 檔案名稱 (pdf_filename)。
#        - 輸出資料夾名稱 (output_folder)。
#
#   2. 檢查輸入檔案與輸出資料夾：
#        - 若 output_folder 不存在則建立。
#        - 若找不到 PDF 檔案則報錯並結束。
#
#   3. PDF → 圖片轉換：
#        - 使用 pdf2image.convert_from_path() 轉換為 PIL Image。
#        - 每頁依序輸出為 PNG，命名為 page_001.png, page_002.png, ...
#
#   4. 完成：
#        - 顯示轉換進度與輸出結果。
# ===================================================================
import os
from pdf2image import convert_from_path

# --- 你的設定 (請修改以下路徑) ---

# 1. Poppler 的 bin 資料夾絕對路徑
#    (這是 poppler-windows 解壓縮後，裡面 Library/bin 的路徑)
poppler_path = r"C:\poppler\poppler-25.07.0\Library\bin"  # <<<<<<<<<<<<< 警告：請務必改成你電腦上 poppler 的真實路徑

# 2. 輸入的 PDF 檔案名稱 (使用相對路徑，假設與此 py 檔在同個資料夾)
pdf_filename = "data.pdf"

# 3. 存放輸出圖片的資料夾名稱
output_folder = "pages"


# --- 主程式 (更穩健的版本) ---

# 檢查輸出資料夾是否存在，若無則建立
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"正在尋找 PDF 檔案：{pdf_filename}")

# 檢查 PDF 檔案是否存在
if not os.path.isfile(pdf_filename):
    print(f"錯誤：在當前資料夾中，找不到指定的 PDF 檔案！ -> {pdf_filename}")
else:
    print(f"開始將 PDF 轉換為圖片，請稍候...")

    try:
        # 將 PDF 的每一頁轉換成圖片
        # 在這裡，我們把 poppler_path 這個參數傳了進去！
        pages = convert_from_path(
            pdf_path=pdf_filename,
            dpi=250,
            thread_count=20,
            poppler_path=poppler_path  # <<<<<<<<<<<<< 就是這一行！
        )

        # 儲存 PNG
        index = 1
        for page in pages:
            output_filename = os.path.join(output_folder, f"page_{index:03d}.png")
            print(f"正在儲存： {output_filename}")
            page.save(output_filename, "PNG")
            index = index + 1

        print(f"\n所有頁面轉換完成！圖片已儲存至 '{output_folder}' 資料夾。")

    except Exception as e:
        print(f"\n轉換過程中發生錯誤！")
        print("請再次確認 Poppler 的路徑是否正確？")
        print(f"詳細錯誤訊息：{e}")