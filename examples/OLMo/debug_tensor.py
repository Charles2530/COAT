import torch
import os

# 替换为你的文件实际路径
file_path = '/mnt/lm_data_afs/wangzining/charles/COAT/examples/OLMo/tensors_Olmo7B/1000/fwd_in_100000.pt'

try:
    # 加载 tensor
    t = torch.load(file_path, map_location='cpu')
    
    print(f"--- Tensor 分析: {os.path.basename(file_path)} ---")
    print(f"1. Shape (形状): {t.shape}")
    print(f"2. Dtype (类型): {t.dtype}")
    print(f"3. Element Count (元素数): {t.numel()}")
    print(f"4. Estimated Size (估算大小): {t.element_size() * t.numel() / 1024 / 1024:.2f} MB")
    print(f"5. Min/Max/Mean: {t.min():.4f}, {t.max():.4f}, {t.mean():.4f}")
    
    # 这一步是为了推测它是什么层
    if t.dim() == 3: 
        print("推测: 可能是 (Batch, SeqLen, HeadDim) -> 单个 Attention Head 的输出")
    elif t.dim() == 4:
        print("推测: 可能是 (Batch, Head, SeqLen, SeqLen) -> Attention Score (Logits)")
    elif t.dim() == 2:
        print("推测: 可能是 (SeqLen, Dim) -> Embedding 或 RoPE 缓存")

except Exception as e:
    print(f"加载失败: {e}")
