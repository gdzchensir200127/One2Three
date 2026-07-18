import open3d as o3d 
import numpy as np
import copy
from signal_simulate.HPR.translate_mesh import translate_mesh
from signal_simulate.HPR.scale_mesh import scale_mesh
from sklearn.neighbors import KDTree

def sample_points(mesh,num_points):
  areas = []
  total_area = 0
  vertices = np.asarray(mesh.vertices)
  for face_vertices in np.asarray(mesh.triangles):
      v1 = vertices[face_vertices[1]] - vertices[face_vertices[0]]
      v2 = vertices[face_vertices[2]] - vertices[face_vertices[0]]
      normal = np.cross(v1, v2)
      normal_length = np.linalg.norm(normal)
      area = 0.5 * normal_length
      areas.append(area)
      total_area += area 
  # 存储采样点的坐标
  points = []

  points_face_index = []

  points_nums = []

  # 遍历每个面
  for i,face in enumerate(mesh.triangles):
    num_face_points = int(num_points*(areas[i]/total_area))

    points_nums.append(num_face_points)
    
    # 获取面的三个顶点的坐标
    face_vertices = np.asarray(mesh.vertices)[face]
    
    # 在面上均匀采样点
    samples = np.random.uniform(0, 1, (num_face_points, 2))
    samples = samples[samples.sum(axis=1) <= 1]  # 仅保留位于面内的采样点
    
    for sample in samples:
        # 计算采样点的坐标
        point = (1 - sample[0] - sample[1]) * face_vertices[0] + \
                sample[0] * face_vertices[1] + \
                sample[1] * face_vertices[2]
        points.append(point)
        points_face_index.append(i)

  # 创建点云数据
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  return pcd,points_face_index,points_nums


def hidden_faces_removal(init_mesh_path:str,output_path:str,display:bool = False,camera=[0, 0.08, 0.60]):
  mesh = o3d.io.read_triangle_mesh(init_mesh_path)
  # Tensor TriangleMesh not supported this function yet.

  # mesh.remove_unreferenced_vertices()
  
  # mesh_dec = mesh.simplify_quadric_decimation(1000)

  # if display:
  #   mesh_dec.compute_vertex_normals()
  #   o3d.visualization.draw_geometries([mesh_dec],window_name = "init mesh")

  # mesh = scale_mesh(mesh)

  # mesh = translate_mesh(mesh)

  pcd,points_face_index,points_nums = sample_points(mesh,len(mesh.triangles)*20)

  if display:
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh],window_name = "init mesh")

  faces = len(mesh.triangles)

  #pcd = mesh.sample_points_poisson_disk(20000)
  if display:
    o3d.visualization.draw_geometries([pcd])

  radius = 50

  new_points_nums = [0] * len(mesh.triangles)
  _, pt_map = pcd.hidden_point_removal(camera, radius)

  for i in pt_map:
     new_points_nums[points_face_index[i]] += 1

  pcd = pcd.select_by_index(pt_map)
  if display:
    o3d.visualization.draw_geometries([pcd])

  # points = pcd.points

  # kd_tree = o3d.geometry.KDTreeFlann()
  # kd_tree.set_geometry(pcd)

  # distances = pcd.compute_nearest_neighbor_distance()
  
  # avg_dist = np.mean(distances)

  # vertices_to_delete = []

  # for i,vertex in enumerate(mesh.vertices):
  #   [k,indices,dis] = kd_tree.search_knn_vector_3d(vertex,10)
  #   if np.mean(dis)>=2*avg_dist:
  #     vertices_to_delete.append(i)

  # mesh.remove_vertices_by_index(vertices_to_delete)

  # mesh.remove_unreferenced_vertices()

  new_triangles = []
  for i,face in enumerate(mesh.triangles):
    if new_points_nums[i]*10 > points_nums[i]:
       new_triangles.append(face)
    # ver1 = mesh.vertices[face[0]]
    # ver2 = mesh.vertices[face[1]]
    # ver3 = mesh.vertices[face[2]]
    # midpoint = (ver1+ver2+ver3)/3
    # [k,indices,dis] = kd_tree.search_knn_vector_3d(midpoint,5)
    # if np.mean(dis)<2*avg_dist:
    #   new_triangles.append(face)

  mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
  mesh.remove_unreferenced_vertices()


  # mesh = mesh.simplify_quadric_decimation(1000)
  
  # radius = 3 * avg_dist

  # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

  # final_mesh = bpa_mesh.simplify_quadric_decimation(10000)

  # final_mesh.remove_degenerate_triangles()
  # final_mesh.remove_duplicated_triangles()
  # final_mesh.remove_duplicated_vertices()
  # final_mesh.remove_non_manifold_edges()

  # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

  # bbox = pcd.get_axis_aligned_bounding_box() 
  # final_mesh = poisson_mesh.crop(bbox)


  mesh.compute_vertex_normals()
  if display:
    o3d.visualization.draw_geometries([mesh],window_name = "mesh after HPR")

  o3d.io.write_triangle_mesh(output_path,mesh)

if __name__ == "__main__":
  #hidden_faces_removal('/home/muxin/hdd/shapenet/sem/models/101354f9d8dede686f7b08d9de913afe.obj',"2",True)
  #hidden_faces_removal('/home/muxin/hdd/shapenet/sem/models/100f39dce7690f59efb94709f30ce0d2.obj',"2",True)

  #hidden_faces_removal("/home/muxin/hdd/shapenet/sem/models/1785410412e01bb9a054360a814c8008.obj","3",True)
  hidden_faces_removal("/home/muxin/hdd/signal_simulate/data/mesh/display.obj","/home/muxin/hdd/signal_simulate/data/mesh/display_pro.obj",True)