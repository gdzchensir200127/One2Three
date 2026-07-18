import numpy as np
import open3d as o3d
import copy
import random

def translate_mesh(mesh,display:bool = False):

  max_bound = mesh.get_max_bound()
  min_bound = mesh.get_min_bound()

  size = max_bound - min_bound

  center = mesh.get_center()

  translate_vector = (-center[0],-min_bound[1],-center[2])

  mesh_copy = copy.deepcopy(mesh)
  mesh_copy.translate(translate_vector)
  

  center = mesh_copy.get_center()
  
  if display:
    mesh_copy.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_copy])
  return mesh_copy

if __name__ == "__main__":
  # mesh = o3d.io.read_triangle_mesh("/home/muxin/hdd/shapenet/sem/models/1a1ec1cfe633adcdebbf11b1629fc16a.obj")
  mesh = o3d.io.read_triangle_mesh("/home/muxin/hdd/signal_simulate/data/mesh/display.obj")
  translate_mesh(mesh,True)
  