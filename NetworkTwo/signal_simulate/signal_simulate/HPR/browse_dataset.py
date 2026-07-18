import os,random
import open3d as o3d

sem_dataset_path = "/home/muxin/hdd/mesh_check"

file_list = []

obj_names = ['0.obj','1.obj','2.obj','3.obj']

def walk_dir(root_dir):
    for dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir,dir)):
            obj_path = os.path.join(root_dir,dir,'models')
            file_list.append(obj_path)

def browse_dataset(root_dir):
    walk_dir(root_dir)
    print("model num: ",len(file_list))
    file_index = 0
    while file_index<len(file_list):
        dir_name = file_list[file_index]
        for obj_name in obj_names:
            file_path = os.path.join(dir_name,obj_name)
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])
            max_bound = mesh.get_max_bound()
            min_bound = mesh.get_min_bound()

            size = max_bound - min_bound
            print("size: ",size)
            center = mesh.get_center()
            print("center",center)
            file_index+=1

if __name__ == "__main__":
    browse_dataset(sem_dataset_path)