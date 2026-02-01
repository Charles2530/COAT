import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
steps = np.arange(0, 1000, 1)
# FP16: 平滑下降
loss_fp16 = 5 * np.exp(-steps/200) + 0.5 + np.random.normal(0, 0.02, len(steps))
# INT4: 下降慢，且后期震荡甚至发散
loss_int4 = 5 * np.exp(-steps/300) + 0.8 + np.random.normal(0, 0.15, len(steps)) 
# 在后期添加发散趋势
loss_int4[600:] += np.linspace(0, 1.5, 400) 

plt.figure(figsize=(8, 5))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(steps, loss_fp16, label='FP16 Baseline (Stable)', color='#1f77b4', linewidth=2)
plt.plot(steps, loss_int4, label='Naive 4-bit (Diverged)', color='#d62728', linestyle='--', linewidth=2, alpha=0.8)

plt.title('Challenge 1: Training Instability in Low-Bit', fontsize=14, fontweight='bold')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加标注箭头
plt.annotate('Gradient Explosion / Noise', xy=(800, 2.5), xytext=(500, 3.5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('test.png')