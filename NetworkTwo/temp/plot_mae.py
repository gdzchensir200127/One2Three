import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import math
plt.style.use('default')  # 重置为默认样式
plt.rcParams.update({
    'axes.facecolor': 'white',   # 坐标区域背景
    'figure.facecolor': 'white', # 画布背景
    'grid.color': 'none',        # 关闭网格线
    'axes.edgecolor': 'black',   # 坐标轴颜色
    'xtick.color': 'black',      # X轴刻度颜色
    'ytick.color': 'black'       # Y轴刻度颜色
})
# 配置参数 ==============================================================
LOG_PATHS = {
    "Fine-tuning": "/home/zhang_muxin/Signal2PC/summary/train_speed/signal2pixel_signal2pixel_4views_encoder_shared_out4_realdataset_unseen_finetune_0408212259/events.out.tfevents.1744118579.node01.2210623.0",  # 替换为微调实验事件文件路径
    "Scratch": "/home/zhang_muxin/Signal2PC/summary/train_speed/signal2pixel_signal2pixel_4views_encoder_shared_out4_realdataset_unseen_0406222003/events.out.tfevents.1743949203.node01.3406226.0"        # 替换为从头训练事件文件路径
}
TAG_NAME = 'eval_depth_dist'            # TensorBoard中记录的标量名称
OUTPUT_IMG = "eval_mae_compare.png"  # 输出图片文件名
DPI = 300                    # 输出图片分辨率
# ======================================================================
def smooth_data(df, method='ewma', alpha=0.6, window=21, gaussian_sigma=3):
    """
    数据平滑处理函数
    参数：
        df: 包含原始数据的DataFrame
        method: 平滑方法 ['ewma', 'moving_avg', 'gaussian']
        alpha: EWMA衰减系数 (0 < alpha <= 1)
        window: 移动平均窗口大小（奇数）
        gaussian_sigma: 高斯核标准差
    返回：
        添加平滑后的DataFrame
    """
    if method == 'ewma':
        # 指数加权移动平均 (TensorBoard默认方法)
        df['smooth'] = df['value'].ewm(alpha=alpha, adjust=False).mean()
        
    elif method == 'moving_avg':
        # 中心化移动平均
        pad = window // 2
        df['padded'] = np.pad(df['value'], (pad,pad), mode='edge')
        df['smooth'] = df['padded'].rolling(window, center=True).mean().iloc[pad:-pad]
        df.drop(columns=['padded'], inplace=True)
        
    elif method == 'gaussian':
        # 高斯滤波
        from scipy.ndimage import gaussian_filter1d
        df['smooth'] = gaussian_filter1d(df['value'], sigma=gaussian_sigma)
        
    else:
        raise ValueError("Unsupported smoothing method")
    
    return df

def load_tb_data(log_path, tag=TAG_NAME):
    """从单个事件文件加载指定tag的数据"""
    ea = EventAccumulator(log_path)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        available_tags = ', '.join(ea.Tags()['scalars'])
        raise ValueError(f"Tag '{tag}' not found. Available tags: {available_tags}")
    
    events = ea.Scalars(tag)
    ret = pd.DataFrame({
        'step': [e.step//2 for i,e in enumerate(events) if i%2 ==0],
        'value': [e.value * 1000 for i,e in enumerate(events) if i%2==0],
        # 'wall_time': [e.wall_time for e in events]
    })
    return ret

# 加载所有实验数据
dfs = []
for exp_name, log_path in LOG_PATHS.items():
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"日志文件不存在: {log_path}")
    
    df = load_tb_data(log_path)
    df['experiment'] = exp_name  # 添加实验类型列
    dfs.append(df)

# 合并数据
combined_df = pd.concat(dfs).reset_index(drop=True)

# 可视化设置
# plt.style.use('seaborn')  # 使用更美观的样式
plt.figure(figsize=(8, 6))

# 为不同实验定义样式
STYLE_CONFIG = {
    "Fine-tuning": {'color': '#FF6B6B', 'linestyle': '-', 'linewidth': 2},
    "Scratch": {'color': '#4ECDC4', 'linestyle': '-', 'linewidth': 2}
}
# combined_df = combined_df.groupby('experiment').apply(
#     lambda x: smooth_data(x, method='ewma', alpha=0.5)
# ).reset_index(drop=True)
# 绘制曲线
for exp_name, group_df in combined_df.groupby('experiment'):
    plt.plot(
        group_df['step'], 
        group_df['value'],
        label=exp_name,
        **STYLE_CONFIG[exp_name]
    )

plt.ylim(bottom=0, top=10)
# 图表装饰
plt.xlabel('Training Steps', fontsize=14, labelpad=10)
plt.ylabel('Eval D-MAE', fontsize=14, labelpad=10)
# plt.title('Training Loss Comparison: Fine-tuning vs Training from Scratch', 
#          fontsize=14, pad=20)
plt.legend(frameon=True, loc='upper right', fontsize=16)
plt.grid(True, alpha=0.3)

# 保存和显示
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=DPI, bbox_inches='tight')
print(f"对比图已保存至: {os.path.abspath(OUTPUT_IMG)}")
plt.show()
