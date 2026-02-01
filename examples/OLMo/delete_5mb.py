import os
import glob
from tqdm import tqdm

# --- 配置 ---
TARGET_DIR = "./tensors_Olmo1B/500"  # 你的 tensor 文件夹路径
SIZE_THRESHOLD_MB = 5                 # 阈值：小于 5MB 的文件统统删掉
# ----------------

def clean_tensors():
    if not os.path.exists(TARGET_DIR):
        print(f"路径不存在: {TARGET_DIR}")
        return

    # 获取所有 .pt 文件
    files = glob.glob(os.path.join(TARGET_DIR, "*.pt"))
    print(f"扫描到 {len(files)} 个文件，准备清洗...")

    deleted_count = 0
    deleted_size = 0

    for file_path in tqdm(files):
        try:
            # 获取文件大小 (字节)
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)

            # 如果小于阈值，删除
            if size_mb < SIZE_THRESHOLD_MB:
                os.remove(file_path)
                deleted_count += 1
                deleted_size += size_mb
        except OSError as e:
            print(f"删除失败 {file_path}: {e}")

    print(f"\n清洗完成！")
    print(f"共删除文件: {deleted_count} 个")
    print(f"释放空间: {deleted_size / 1024:.2f} GB")

if __name__ == "__main__":
    # 再次确认防止误删
    print(f"警告：将删除 {TARGET_DIR} 下所有小于 {SIZE_THRESHOLD_MB}MB 的文件。")
    confirm = input("确认请输入 'yes': ")
    if confirm == "yes":
        clean_tensors()
    else:
        print("取消操作")
