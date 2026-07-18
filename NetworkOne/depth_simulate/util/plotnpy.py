import numpy as np
import matplotlib.pyplot as plt


def plot_npy_heatmap(file_path):
    """
    读取npy数组并使用matplotlib绘制热力图，支持鼠标悬停显示坐标和值

    参数:
        file_path: npy文件的路径
    """
    # 读取npy数组
    try:
        data = np.load(file_path)
        print(f"成功读取数据，形状: {data.shape}")
        print(f"原始数据类型: {data.dtype}")
        data = data[0,:,:,0]
        data = np.abs(data)
        print(f"取绝对值后的数据类型: {data.dtype}")

        # 验证数组形状是否为2880×96
        if data.shape != (2880, 96):
            raise ValueError(f"数组形状不符合要求！预期(2880, 96)，实际{data.shape}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return
    except Exception as e:
        print(f"读取数据时发生错误: {str(e)}")
        return

    # 设置中文字体，确保中文正常显示
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 20))  # (宽度, 高度)，单位为英寸

    # 绘制热力图
    im = ax.imshow(
        data,
        cmap='viridis',  # 颜色映射
        aspect='auto'  # 自动调整宽高比
    )

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('数值', rotation=270, labelpad=20)

    # 设置标题和轴标签
    ax.set_title('2880×96数组热力图（鼠标悬停查看详情）', fontsize=16, pad=20)
    ax.set_xlabel('Bin', fontsize=12)
    ax.set_ylabel('Frame', fontsize=12)

    # 优化刻度显示
    ax.set_xticks(np.arange(0, 96, 10))
    ax.set_xticklabels(np.arange(0, 96, 10), fontsize=8)
    ax.set_yticks(np.arange(0, 2880, 200))
    ax.set_yticklabels(np.arange(0, 2880, 200), fontsize=8)

    # 创建一个文本标注，用于显示鼠标悬停时的信息
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # 鼠标移动事件处理函数
    def update_annot(event):
        if event.inaxes == ax:
            # 获取鼠标位置对应的数组坐标
            x_idx = int(round(event.xdata))
            y_idx = int(round(event.ydata))

            # 检查坐标是否在数据范围内
            if 0 <= x_idx < data.shape[1] and 0 <= y_idx < data.shape[0]:
                # 获取该点的值
                value = data[y_idx, x_idx]

                # 更新标注位置和内容
                annot.xy = (x_idx, y_idx)
                text = f'Frame: {y_idx}\nBin: {x_idx}\n值: {value:.6f}'
                annot.set_text(text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                # 如果鼠标在数据范围外，隐藏标注
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    # 绑定鼠标移动事件
    fig.canvas.mpl_connect("motion_notify_event", update_annot)

    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()


if __name__ == "__main__":
    # 请替换为你的npy文件路径
    npy_file_path = "E:/16_扳手_1.npy"
    plot_npy_heatmap(npy_file_path)
