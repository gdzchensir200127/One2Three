import warnings
import numpy as np

import trimesh
import pyrender
import time


def depth_simulate(
        merged_vertices: list[list[float]],
        merged_faces: list[list[int]],
        camera_points: list[float],
        init_place_points: list[float]
) -> np.ndarray:
    """
    使用 trimesh + pyrender 模拟某一帧的深度图，分辨率 640x576。

    :param merged_vertices: 顶点列表 [[x, y, z, ...], ...]
    :param merged_faces: 面列表 [[count, v1, v2, v3], ...]
    :param camera_points: 相机位置 [x, y, z]
    :param init_place_points: 初始放置点位置 [x, y, z]
    :return: CuPy array of shape (576, 640)
    """
    start_time = time.time()
    vertices_np = []  # 顶点坐标 (N, 3)
    for line in merged_vertices:
        if isinstance(line, list):
            line = ' '.join(map(str, line))  # 如果 line 是列表，将其转换为字符串
        parts = list(map(float, line.strip().split()))
        vertices_np.append(parts[:3])  # 坐标
    vertices_np = np.array(vertices_np, dtype=np.float32)
    # # 转换为 NumPy 数组
    # vertices_np = np.array([v[:3] for v in merged_vertices], dtype=np.float32)
    faces_np = np.array([f[1:] for f in merged_faces], dtype=np.int32)

    camera_pos = np.array(camera_points, dtype=np.float32)
    place_pos = np.array(init_place_points, dtype=np.float32)

    # 创建 mesh
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)

    # 构造相机矩阵：
    z_axis = place_pos - camera_pos
    # 投影到 XY 平面
    z_axis[2] = 0.0

    if np.linalg.norm(z_axis) < 1e-6:
        z_axis = np.array([0, 0, 1], dtype=np.float32)
    else:
        z_axis = z_axis / np.linalg.norm(z_axis)

    # 构造 x/y 轴
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(z_axis, up)) > 0.99:
        up = np.array([0, 1, 0], dtype=np.float32)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 构造相机变换矩阵（世界到相机）
    cam_matrix = np.eye(4)
    cam_matrix[:3, 0] = x_axis
    cam_matrix[:3, 1] = y_axis
    cam_matrix[:3, 2] = z_axis
    cam_matrix[:3, 3] = camera_pos

    # 创建场景
    scene = pyrender.Scene()
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3, 1.0])
    mesh_py = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh_py)

    # 添加相机
    camera = pyrender.PerspectiveCamera(yfov=5*np.pi / 12.0, aspectRatio=640 / 576)
    scene.add(camera, pose=cam_matrix)

    # 添加默认光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light)

    # 渲染器
    r = pyrender.OffscreenRenderer(640, 576)
    color, depth = r.render(scene)

    # 关闭渲染器释放资源
    r.delete()

    # 将深度图转为 numPy 数组
    depth_np = np.array(depth, dtype=np.float32)

    total_time = time.time() - start_time
    print(f"深度图生成完成，耗时: {total_time:.3f} 秒")

    return depth_np

