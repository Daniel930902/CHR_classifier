# 📘 CHR_Classifier

[English Version](README.md)

## 📖 簡介
CHR Classifier 是一個基於 OCR 的流程系統，專門用於從補習班掃描的生字練習簿中，辨識並擷取 **繁體中文手寫字跡**。 


此系統會自動處理掃描影像，偵測格子結構，利用 OCR 與白名單推斷進行分類，並將切割出的手寫字樣儲存為資料集。  

---

## 🏫 專案背景
本專案最初開發於 **WASN 實驗室補習班計畫**，該計畫提供了生字練習簿的掃描檔。  


系統的用途是作為 **資料標註與分類工具**，以蒐集常見繁體中文字，並用於後續研究。  

⚠️ **注意**：由於計畫限制，原始的補習班掃描資料 **無法公開**。  
若要使用本專案，請自行準備 **自有的掃描練習簿或文件** 作為輸入，敬請見諒。  

- 目前 **格子偵測與覆蓋率**：**99.99%**  
- 目前 **OCR 分類準確率**：約 **95%**  
- OCR 引擎：**TesseractOCR**（針對繁體中文進行 fine-tune）  

---

## ✨ 功能特色


- 🧩 **格子偵測**：多通道方法（輪廓分析、霍夫變換、投影剖面）。  
- 🔍 **嚴格空白檢測**：利用多重特徵品質控管（持久度遮罩、邊緣密度、連通元件），避免存下空白或雜訊格子。  
- 📝 **白名單推斷**：支援字元序列，自訂起始錨點。  
- 📊 **完整統計報告**：包含儲存率、資料產出率、低存量欄位紀錄。  
- ⚡ **自動化流程**：若需要會自動執行預處理（`pdf2png.py`, `preprocess_pages.py`）。  

---

## 🗂 檔案結構

CHR_classifier

├── CHR_classifier.py # 主 OCR 流程

├── debug_grid.py # 格子偵測除錯工具

├── pdf2png.py # PDF 轉 PNG 工具

├── preprocess_pages.py # 前處理工具

├── whitelist.txt # 字元白名單

├── data/ # 原始輸入的 PNG 頁面

├── pdf/ # 原始輸入的 PDF 檔案

└── datasets/ # 儲存輸出結果的資料夾


---

## 🔧 系統需求
- Python 3.8+
- [OpenCV](https://opencv.org/)
- NumPy
- [PyTesseract](https://github.com/madmaze/pytesseract)  

  ⚠️ **重要**：必須在本地安裝 **Tesseract OCR**，並於程式中設定正確路徑，例如：  

  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

---

## 🚀 使用方法

1. 將掃描後的 PNG 頁面放入 data/cramschool_merged/。

（⚠️ 因原始補習班資料不公開，請使用 自有資料）

2. 編輯 whitelist.txt，填入目標字元。

3. 執行：

python CHR_classifier.py

4. 切割後的手寫字跡圖片與偵測可視化結果將會輸出到指定資料夾。

---

## 🔄 處理流程

```mermaid

flowchart TD
    A[掃描的 PDF/PNG 頁面] --> B[前處理 (pdf2png, preprocess_pages)]
    B --> C[格子偵測 (輪廓 / 霍夫 / 投影)]
    C --> D[標籤列 OCR + 白名單推斷]
    D --> E[動態空白檢查 (多重特徵)]
    E --> F[切割並儲存手寫字跡]
    F --> G[統計報告]

```

---

## 📊 範例輸出

* 按字元整理的手寫字跡圖片

* 標籤 OCR 與格子偵測的除錯圖

* 最終統計報告，包含儲存率與產出率

---

## 🙏 致謝

本專案使用了針對繁體中文 fine-tune 的 Tesseract 模型：
[ tessdata_chi ]( gumblex/tessdata_chi )

---

## 📝 備註
此 README 在 ChatGPT 5 協助下撰寫與完成。