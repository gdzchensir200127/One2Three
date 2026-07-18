import numpy as np
import open3d as o3d
import copy
import random

size_range = (0.2,0.5)

def scale_mesh_factor(mesh,scale_factor,center=np.array([0.,0.,0.]),display:bool = False):
  mesh_copy = copy.deepcopy(mesh)
  mesh_copy.scale(scale_factor,center = center)
  return mesh_copy

def scale_mesh(mesh,display:bool = False):

  max_bound = mesh.get_max_bound()
  min_bound = mesh.get_min_bound()

  size = max_bound - min_bound

  
  axis = np.argmax(size)

  max_size = size[axis]

  center = mesh.get_center()

  if max_size<size_range[0] or max_size>size_range[1]:
    target_size = random.uniform(size_range[0],size_range[1])
    scale_factor = target_size / max_size
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy.scale(scale_factor,center = mesh.get_center())
    
    max_bound = mesh_copy.get_max_bound()
    min_bound = mesh_copy.get_min_bound()

    size = max_bound - min_bound

    center = mesh_copy.get_center()
    
    if display:
      mesh_copy.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_copy])
    return mesh_copy
  
  return mesh


if __name__ == "__main__":
  mesh = o3d.io.read_triangle_mesh("/home/muxin/hdd/shapenet/sem/models/1a1ec1cfe633adcdebbf11b1629fc16a.obj")
  scale_mesh(mesh,True)