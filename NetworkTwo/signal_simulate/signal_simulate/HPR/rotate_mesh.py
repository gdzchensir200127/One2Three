import open3d as o3d 
import numpy as np
import copy

def rotate_mesh(mesh,xyz:tuple,center=(0,0,0),display:bool = False):
  '''
  rotate the mesh and return the mesh after rotate
  mesh: the init mesh
  xyz: the rotate angle of x,y,z axis such as (np.pi/2,0,0)
  center: the center of the rotation
  '''

  # mesh.compute_vertex_normals()

  if display:
    o3d.visualization.draw_geometries([mesh],window_name = "init mesh")

  R = mesh.get_rotation_matrix_from_xyz(xyz)

  mesh_copy = copy.deepcopy(mesh)

  mesh_copy.rotate(R,center = center)
  if display:
    o3d.visualization.draw_geometries([mesh_copy],window_name = "mesh after rotate")

  return mesh_copy


def rotate_mesh_write(init_mesh_path:str,output_path:str,xyz:tuple,center=(0,0,0),display:bool = False):
  '''
  rotate the mesh and save the mesh after rotate to obj
  init_mesh_path: the file path of the init mesh
  output_path: the file path to save the mesh after rotate
  xyz: the rotate angle of x,y,z axis such as (np.pi/2,0,0)
  center: the center of the rotation
  '''
  mesh = o3d.io.read_triangle_mesh(init_mesh_path)

  mesh.compute_vertex_normals()

  if display:
    o3d.visualization.draw_geometries([mesh],window_name = "init mesh")

  R = mesh.get_rotation_matrix_from_xyz(xyz)

  mesh_copy = copy.deepcopy(mesh)

  mesh_copy.rotate(R,center = center)

  if display:
    o3d.visualization.draw_geometries([mesh_copy],window_name = "mesh after rotate")

  o3d.io.write_triangle_mesh(output_path, mesh_copy)

if __name__ == "__main__":
  rotate_mesh("/home/muxin/hdd/signal_simulate/data/mesh/display.obj","/home/muxin/hdd/signal_simulate/data/mesh/display_rotate.obj")