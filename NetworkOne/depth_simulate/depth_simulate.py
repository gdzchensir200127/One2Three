import numpy as np
import trimesh
from trimesh.ray import ray_pyembree
import time


def depth_simulate(
        merged_vertices: np.ndarray,
        merged_faces: np.ndarray,
        camera_points: np.ndarray,
        init_place_points: np.ndarray
) -> np.ndarray:
    """
    使用 trimesh + pyembree 快速模拟深度图，分辨率 640x576。

    :param merged_vertices: 顶点数组，shape=(N, 3+)，前3列为坐标 [x, y, z, ...]
    :param merged_faces: 面数组，shape=(M, 4)，格式为 [count, v1, v2, v3]（count=3）
    :param camera_points: 相机位置数组，shape=(3,)，[x, y, z]
    :param init_place_points: 初始放置点位置数组，shape=(3,)，[x, y, z]
    :return: 深度图数组，shape=(576, 640)，dtype=np.float32
    """
    start_time = time.time()

    # 提取顶点坐标（前3列），直接使用数组切片
    vertices_np = merged_vertices[:, :3].astype(np.float32)

    faces_np = merged_faces[:, 1:4].astype(np.int32)  # 取后3个顶点索引

    # 验证有效面数量
    if faces_np.size == 0:
        raise ValueError("没有有效的三角形面数据，无法生成深度图")

    # 相机和放置点位置确保为float32数组
    camera_pos = camera_points.astype(np.float32)
    place_pos = init_place_points.astype(np.float32)

    # 构建相机坐标系（向量化运算）
    z_axis = place_pos - camera_pos
    z_axis[2] = 0.0  # 投影到XY平面
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0, 0, 1], dtype=np.float32)
    else:
        z_axis /= z_norm  # 归一化

    # 计算上方向和相机轴系
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(z_axis, up)) > 0.99:
        up = np.array([0, 1, 0], dtype=np.float32)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 创建三角网格模型（直接传入数组）
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)

    # 设置图像参数
    width, height = 640, 576
    fov_y = 5 * np.pi / 12.0  # 75度
    aspect_ratio = width / height
    focal_length = height / (2 * np.tan(fov_y / 2))

    # 构造像素坐标网格（向量化生成）
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # 计算射线方向（相机局部坐标 -> 世界坐标）
    x = (u - width / 2) / focal_length
    y = (height / 2 - v) / focal_length

    # 射线方向转换到世界坐标系（利用广播机制）
    ray_directions = (
            x[..., None] * x_axis +
            y[..., None] * y_axis +
            z_axis
    )
    ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)

    # 射线起点（所有射线从相机位置发出，广播为与射线方向同形状）
    ray_origins = np.broadcast_to(camera_pos, ray_directions.shape)

    # 使用pyembree进行射线相交检测（展平为二维数组提高效率）
    intersector = ray_pyembree.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins.reshape(-1, 3),
        ray_directions=ray_directions.reshape(-1, 3)
    )

    # 初始化深度图并填充深度值
    depth_map = np.zeros((height, width), dtype=np.float32)
    if len(locations) > 0:
        # 计算每个交点到相机的距离（向量化计算）
        t_values = np.linalg.norm(locations - camera_pos, axis=1)
        depth_map.flat[index_ray] = t_values  # 利用flat索引赋值

    # 打印耗时
    total_time = time.time() - start_time
    print(f"深度图生成完成，耗时: {total_time:.3f} 秒")

    return depth_map
