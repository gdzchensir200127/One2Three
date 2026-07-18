import open3d as o3d 
import numpy as np
import copy


mesh = o3d.io.read_triangle_mesh("/home/muxin/hdd/shapenet/core/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj")
# Tensor TriangleMesh not supported this function yet.
mesh.compute_vertex_normals()


o3d.visualization.draw_geometries([mesh],window_name = "init mesh")

max_bound = mesh.get_max_bound()
min_bound = mesh.get_min_bound()

size = max_bound - min_bound
print("size: ",size)
center = mesh.get_center()
print("center",center)