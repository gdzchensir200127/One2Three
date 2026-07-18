import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
TAG_NAME = 'loss'            # TensorBoard中记录的标量名称
OUTPUT_IMG = "loss_compare.png"  # 输出图片文件名
DPI = 300                    # 输出图片分辨率
# ======================================================================

def load_tb_data(log_path, tag=TAG_NAME):
    """从单个事件文件加载指定tag的数据"""
    ea = EventAccumulator(log_path)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        available_tags = ', '.join(ea.Tags()['scalars'])
        raise ValueError(f"Tag '{tag}' not found. Available tags: {available_tags}")
    
    events = ea.Scalars(tag)
    return pd.DataFrame({
        'step': [e.step for e in events],
        'value': [e.value for e in events],
        'wall_time': [e.wall_time for e in events]
    })

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
# 添加指数移动平均曲线（可选）
combined_df['smooth'] = combined_df.groupby('experiment')['value'].transform(lambda x: x.ewm(span=50).mean())
# 绘制曲线
for exp_name, group_df in combined_df.groupby('experiment'):
    plt.plot(
        group_df['step'], 
        group_df['value'],
        label=exp_name,
        **STYLE_CONFIG[exp_name]
    )


# 图表装饰
plt.xlabel('Training Steps', fontsize=14, labelpad=10)
plt.ylabel('Loss Value', fontsize=14, labelpad=10)
# plt.title('Training Loss Comparison: Fine-tuning vs Training from Scratch', 
#          fontsize=14, pad=20)
plt.legend(frameon=True, loc='upper right', fontsize=16)
plt.grid(True, alpha=0.3)

# 保存和显示
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=DPI, bbox_inches='tight')
print(f"对比图已保存至: {os.path.abspath(OUTPUT_IMG)}")
plt.show()
