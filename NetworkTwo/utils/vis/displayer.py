import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL.Image as Image

class PtDisplayer(object):
  def __init__(self, xlim: list, ylim: list, zlim: list, color: str = 'b', marker:str = 'o'):
    self.xlim = xlim
    self.ylim = ylim
    self.zlim = zlim
    self.color = color
    self.marker = marker

  def display_points(self, point_cloud):
    # 假设点云数据是一个N*3的NumPy数组或Tensor

    # 创建一个3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取点云数据的坐标
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.set_xlim([self.xlim[0], self.xlim[1]])
    ax.set_ylim([self.ylim[0], self.ylim[1]])
    ax.set_zlim([self.zlim[0], self.zlim[1]])

    # 绘制点云
    ax.scatter(x, y, z, c=self.color, marker=self.marker)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # fig.show()

    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)

    image = np.transpose(image[:,:,:3], (2, 0, 1))

    # fig.canvas.draw()
    # image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return image


  def visualize_reconstruction(self, gt_coord, coord):
    gt_pc = self.display_points(gt_coord)

    pred_pc = self.display_points(coord)
    
    return np.concatenate((gt_pc, pred_pc), 2)


  def s2p_batch_visualize(self, batch_input, batch_output, atmost=3):
    batch_size = min(batch_input["points"].size(0), atmost)
    images_stack = []
    for i in range(batch_size):
        gt_points = batch_input["points"][i].cpu().numpy()
        coord = batch_output['pred_coord'][i].cpu().numpy()
        images_stack.append(self.visualize_reconstruction(gt_points,coord))
    return torch.from_numpy(np.concatenate(images_stack, 1))