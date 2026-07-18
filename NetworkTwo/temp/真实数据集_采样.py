import os
import trimesh
import numpy as np
input_dir ="/home/zhang_muxin/data/obj"
list = ["笔记本电脑","显示器","扳手","手机","刀具","键盘","易拉罐","水杯"]
for item in list:
    for i in range(1,7):
        input_path = os.path.join(input_dir, item, str(i),"models", "model_normalized.obj")
        output_path = os.path.join("/home/zhang_muxin/data/real_dataset_base",item+"_depth_fixed4",str(i)+".ply")
        output_path_2 = os.path.join("/home/zhang_muxin/data/real_dataset_base",item+"-遮挡_depth_fixed4",str(i)+".ply")
        base_dir = os.path.dirname(output_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # 1. 加载OBJ模型
        mesh = trimesh.load(input_path)  # 替换为你的输入文件路径
        num_points = 20000  # 设置采样点数
        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        # 3. 创建点云对象并保存
        point_cloud = trimesh.PointCloud(points)
        point_cloud.export(output_path)  # 支持.ply/.xyz/.pts等格式
        point_cloud.export(output_path)

