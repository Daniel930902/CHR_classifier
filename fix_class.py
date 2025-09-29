# -*- coding: utf-8 -*-
# ===================================================================
#
#   fix_class.py
#   用來手動檢查並重新整理及編號的程式
#
#   核心修改:
#   1. 新增功能：在程式啟動時，讓使用者手動輸入要處理的主資料夾名稱。
#   2. 修正並優化 renumber_folder 函式中的兩階段改名邏輯。
#
# ===================================================================
import os
import shutil
import re

def get_max_file_number(folder_path):
    """查找指定資料夾中，所有 '數字.png' 格式檔案的最大編號。"""
    if not os.path.isdir(folder_path):
        return 0
    max_num = 0
    pattern = re.compile(r'^(\d+)\.png$')
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num

def renumber_folder(folder_path):
    """
    對指定資料夾內的檔案 (001.png, 002.png...) 重新進行連續編號。
    採用安全的「兩階段改名法」避免檔案衝突。
    """
    print(f"\n正在重新編號資料夾 '{folder_path}'...")
    
    try:
        if not os.path.isdir(folder_path):
            print(f"錯誤：找不到資料夾 '{folder_path}'。")
            return
            
        files_to_renumber = [f for f in os.listdir(folder_path) if re.match(r'^\d+\.png$', f)]
        files_to_renumber.sort(key=lambda f: int(f.split('.')[0]))
        
        if not files_to_renumber:
            print("✔ 資料夾中沒有需要重新編號的檔案。")
            return
            
    except FileNotFoundError:
        print(f"錯誤：找不到資料夾 '{folder_path}'。")
        return
    
    temp_suffix = "_temp_rename.png"
    temp_files = []

    # --- 階段一：全部改為暫存檔名 ---
    for old_filename in files_to_renumber:
        old_filepath = os.path.join( folder_path, old_filename)
        # 移除 .png 再加上後綴，避免檔名變成 001.png_temp_rename
        base_name = old_filename.split('.')[0]
        temp_filepath = os.path.join(folder_path, base_name + temp_suffix)
        try:
            os.rename(old_filepath, temp_filepath)
            temp_files.append(base_name + temp_suffix)
        except Exception as e:
            print(f"  錯誤：暫時改名 {old_filename} 失敗。原因: {e}")
            # 如果失敗，可能需要手動介入，此處先跳過
            continue
    
    # 確保暫存檔案列表也經過數字排序
    temp_files.sort(key=lambda f: int(f.split('_')[0]))

    # --- 階段二：從暫存檔名改回最終序列檔名 ---
    renamed_count = 0
    for i, temp_filename in enumerate(temp_files):
        new_number = i + 1
        new_filename = f"{new_number:03d}.png"
        
        temp_filepath = os.path.join(folder_path, temp_filename)
        new_filepath = os.path.join(folder_path, new_filename)
        
        try:
            print(f"  -> 正在改名: {temp_filename} -> {new_filename}")
            os.rename(temp_filepath, new_filepath)
            renamed_count += 1
        except Exception as e:
            print(f"  錯誤：從暫存檔改名失敗 {temp_filename}。原因: {e}")

    if renamed_count > 0:
        print(f"✔ 資料夾重新編號完成，共處理了 {renamed_count} 個檔案。")
    else:
        print("✔ 資料夾檔案序列無需重新編號。")


def main():
    """主執行函式"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ========================= 【修改後】 =========================
    while True:
        base_dir = input("\n请输入要處理的【主資料夾完整路徑】 (例如: E:\\datasets 或 C:\\Users\\ASUS\\cram_school_scan): ").strip()
        if not base_dir:
            print("錯誤：主資料夾路徑不能為空。")
            continue
        if not os.path.isdir(base_dir):
            print(f"錯誤：找不到資料夾 '{base_dir}'，請確認路徑是否正確。")
        else:
            print(f"✔ 目標主資料夾設定為: '{base_dir}'")
            break
    # ===========================================================

    while True:
        print("\n" + "=" * 50)
        print("請輸入下一個要校正的任務 (或輸入 'q' 退出)")
        
        source_char = input(f"请输入【來源】資料夾的標籤字: ")
        if source_char.lower() == 'q':
            print("感謝使用，程式已結束。")
            break
        source_dir = os.path.join(base_dir, source_char)
        if not os.path.isdir(source_dir):
            print(f"錯誤：在 '{base_dir}' 中找不到資料夾 '{source_char}'，請重新開始。")
            continue
        
        dest_char = input(f"请输入【目的地】資料夾的標籤字 (若只想整理來源資料夾，請直接按 Enter): ")
        
        if not dest_char:
            renumber_folder(source_dir)
            continue
        
        if dest_char.lower() == 'q':
            print("感謝使用，程式已結束。")
            break

        dest_dir = os.path.join(base_dir, dest_char)
        if not os.path.isdir(dest_dir):
            print(f"提示：目的地資料夾 '{dest_dir}' 不存在，將會自動為您建立。")
        
        # ... (後續的移動邏輯不變) ...
        print("-" * 40)
        print(f"來源：{source_dir}")
        print(f"目的：{dest_dir}")
        print("-" * 40)

        files_to_move = []
        source_files = sorted([f for f in os.listdir(source_dir) if re.match(r'^\d+\.png$', f)], 
                              key=lambda f: int(f.split('.')[0]))

        start_input = input("请输入起始編號 (直接按 Enter 表示全選): ")

        if not start_input:
            files_to_move = source_files
            if not files_to_move:
                print("錯誤：來源資料夾中沒有任何圖片檔案可移動。")
                continue
            print(f"✔ 已選擇全部 {len(files_to_move)} 個檔案。")
        else:
            try:
                end_input = input("请输入結束編號 (直接按 Enter 表示只移動單張): ")
                start = int(start_input)
                end = start if not end_input else int(end_input)
                
                if start > end:
                    print("錯誤：起始編號不能大於結束編號。"); continue
                
                for i in range(start, end + 1):
                    filename = f"{i:03d}.png"
                    if filename in source_files:
                        files_to_move.append(filename)
                    else:
                        print(f"警告：在來源資料夾中找不到檔案 '{filename}'，已跳過。")
                
                if not files_to_move:
                    print("錯誤：在指定範圍內沒有找到任何檔案。"); continue
                
                if start == end:
                    print(f"✔ 已選擇單張檔案 (編號: {start})。")
                else:
                    print(f"已選擇 {len(files_to_move)} 個檔案 (從 {start} 到 {end})。")
            except ValueError:
                print("錯誤：請輸入有效的數字。"); continue

        print("-" * 40)
        os.makedirs(dest_dir, exist_ok=True)
        start_num = get_max_file_number(dest_dir)
        print(f"目的地資料夾 '{dest_dir}' 目前最大編號為 {start_num}。")
        print(f"將從 {start_num + 1:03d}.png 開始重新命名...")

        for i, filename in enumerate(files_to_move):
            old_path = os.path.join(source_dir, filename)
            new_number = start_num + 1 + i
            new_filename = f"{new_number:03d}.png"
            new_path = os.path.join(dest_dir, new_filename)
            print(f"正在移動: {old_path} -> {new_path}")
            shutil.move(old_path, new_path)
            
        print(f"🎉 校正完成！成功移動了 {len(files_to_move)} 個檔案。")
        
        renumber_folder(source_dir)
        
        try:
            if not os.listdir(source_dir):
                print(f"提示：來源資料夾 '{source_dir}' 現已清空，將自動刪除...")
                os.rmdir(source_dir)
                print(f"✔ 已成功刪除資料夾 '{source_dir}'。")
        except OSError as e:
            print(f"錯誤：嘗試刪除資料夾 '{source_dir}' 失敗。原因: {e}")

if __name__ == '__main__':
    main()
