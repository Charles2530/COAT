import os
import hashlib
import random
import sys
import time

# ================= 配置区域 =================
TARGET_DIR = 'newtensors_Olmo1B/500'  # 你的目标文件夹
SAMPLE_COUNT = 10                     # 随机抽样的数量
OUTPUT_FILE = 'duplicate_report.txt'  # 结果保存文件
BLOCK_SIZE = 65536                    # 哈希读取块大小 (64KB)
# ===========================================

def get_file_hash(filepath):
    """计算文件MD5，带异常处理"""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(BLOCK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def print_progress(current, total, start_time):
    """打印进度条"""
    percent = float(current) * 100 / total
    elapsed_time = time.time() - start_time
    
    # 估算剩余时间
    if current > 0:
        avg_time_per_item = elapsed_time / current
        remaining_items = total - current
        remaining_time = remaining_items * avg_time_per_item
        rem_str = f"{int(remaining_time)}s"
    else:
        rem_str = "?"

    # 进度条长度 30 字符
    arrow = '-' * int(percent/100 * 30 - 1) + '>'
    spaces = ' ' * (30 - len(arrow))
    
    # \r 让光标回到行首，实现动态刷新
    sys.stdout.write(f"\r进度: [{arrow}{spaces}] {percent:.1f}% | 已处理: {current}/{total} | 预计剩余: {rem_str}")
    sys.stdout.flush()

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"错误：找不到路径 {TARGET_DIR}")
        return

    print("1. 正在读取文件列表（请稍候）...")
    try:
        all_files = [f for f in os.listdir(TARGET_DIR) if f.endswith('.pt')]
    except Exception as e:
        print(f"读取目录失败: {e}")
        return

    total_files = len(all_files)
    if total_files == 0:
        print("目录中没有 .pt 文件。")
        return

    # 随机抽样
    real_sample_count = min(total_files, SAMPLE_COUNT)
    sample_names = random.sample(all_files, real_sample_count)
    
    print(f"2. 正在分析 {real_sample_count} 个样本特征...")
    
    # 数据结构：
    # target_db key=Hash, value={ 'original_name': str, 'size': int, 'paths': [] }
    target_db = {}
    target_sizes = set() # 用于快速通过大小过滤
    
    for name in sample_names:
        path = os.path.join(TARGET_DIR, name)
        size = os.path.getsize(path)
        f_hash = get_file_hash(path)
        
        if f_hash:
            target_db[f_hash] = {
                'original_name': name,
                'size': size,
                'paths': [] # 稍后用来存所有找到的相同文件的路径
            }
            target_sizes.add(size)

    print(f"3. 开始全盘扫描比对 (共 {total_files} 个文件)...")
    print("   (策略：仅当文件大小完全一致时，才读取内容计算哈希)")
    
    start_time = time.time()
    
    # === 主循环 ===
    for i, fname in enumerate(all_files):
        # 打印进度条 (每处理 10 个文件刷新一次显示，避免刷屏太快影响性能)
        if i % 10 == 0:
            print_progress(i + 1, total_files, start_time)
            
        fpath = os.path.join(TARGET_DIR, fname)
        try:
            # 第一层过滤：文件大小
            f_size = os.path.getsize(fpath)
            if f_size in target_sizes:
                # 第二层过滤：哈希比对
                f_hash = get_file_hash(fpath)
                if f_hash in target_db:
                    # 找到一个相同文件（包含样本自身）
                    target_db[f_hash]['paths'].append(fname)
        except OSError:
            continue
            
    # 循环结束，打印满格进度
    print_progress(total_files, total_files, start_time)
    print("\n\n扫描完成！正在写入报告...")

    # === 写入文件 ===
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"扫描报告 - 目标目录: {TARGET_DIR}\n")
        f.write(f"扫描时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # 按重复数量排序
        sorted_results = sorted(target_db.values(), key=lambda x: len(x['paths']), reverse=True)
        
        for item in sorted_results:
            count = len(item['paths'])
            f.write(f"● 样本文件: {item['original_name']}\n")
            f.write(f"  文件大小: {item['size'] / (1024*1024):.2f} MB\n")
            f.write(f"  相同文件数量: {count} (含自身)\n")
            f.write(f"  文件列表:\n")
            for p in item['paths']:
                f.write(f"    - {p}\n")
            f.write("-" * 50 + "\n")

    # === 屏幕简报 ===
    print("-" * 60)
    print(f"{'样本文件名':<35} | {'相同数量'}")
    print("-" * 60)
    for item in sorted_results:
        print(f"{item['original_name']:<35} | {len(item['paths'])}")
    print("-" * 60)
    print(f"详细路径列表已保存至: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
