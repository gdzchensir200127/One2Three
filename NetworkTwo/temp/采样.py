import os
import trimesh
import numpy as np
input_root_dir ="/home/zhang_muxin/data/shapenet_obj"
list = ["02933112","03636649","04090263","04379243","04530566","02828884","03211117","03691459","04256520","04401088","02691156","02958343","03001627"]
for item in list:
    input_dir = os.path.join(input_root_dir, item)
    dir_list = os.listdir(input_dir)
    for dir_ in dir_list:
        input_path = os.path.join(input_dir, dir_,"models", "model_normalized.obj")
        output_path = os.path.join("/home/zhang_muxin/data/shapenet_dataset_final",item+"_depth_fixed4",dir_+".ply")
        base_dir = os.path.dirname(output_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # 1. 加载OBJ模型
        mesh = trimesh.load(input_path)  # 替换为你的输入文件路径
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        num_points = 10000  # 设置采样点数
        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        # 3. 创建点云对象并保存
        point_cloud = trimesh.PointCloud(points)
        point_cloud.export(output_path)  # 支持.ply/.xyz/.pts等格式

