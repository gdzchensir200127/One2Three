import open3d as o3d 
import numpy as np
import copy
from signal_simulate.util.plot import plot_radar_data
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from signal_simulate.HPR.hidden_faces_removal import hidden_faces_removal

# init_mesh = o3d.io.read_triangle_mesh("/home/muxin/hdd/shapenet/core/ShapeNetCore.v2/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/model_normalized.obj")
# # Tensor TriangleMesh not supported this function yet.
# init_mesh.compute_vertex_normals()


# o3d.visualization.draw_geometries([init_mesh],window_name = "init mesh")

camera = [0,0.5,0.6]
hidden_faces_removal("/home/muxin/hdd/shapenet/core/ShapeNetCore.v2/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/model_normalized.obj","/tmp/tmp.obj",True,camera)

# hpr_mesh = o3d.io.read_triangle_mesh("/home/muxin/hdd/shapenetS2M_HPR/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/0.obj")
# # Tensor TriangleMesh not supported this function yet.
# hpr_mesh.compute_vertex_normals()

# o3d.visualization.draw_geometries([hpr_mesh],window_name = "mesh after HPR")

# signal_path = "/home/muxin/hdd/shapenetS2M_signal/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/0.mat"
# signal = loadmat(signal_path)['data']

# fig1,ax1 = plt.subplots(1,1)

# plot_radar_data(ax1,signal,'simulat signal')

# plt.show()

cut_signal_path = "/home/muxin/hdd/shapenetS2M/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/models/0.mat"
cut_signal = loadmat(cut_signal_path)['data']

fig2,ax2 = plt.subplots(1,1)

plot_radar_data(ax2,cut_signal,'simulat signal after cut')

plt.show()
