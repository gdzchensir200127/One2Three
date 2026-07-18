import numpy as np
import pyvista as pv

def depth_simulate(
        merged_vertices: list[list[float]],
        merged_faces: list[list[int]],
        camera_points: list[float],
        init_place_points: list[float]
) -> np.ndarray:
    """
    使用 PyVista 模拟某一帧的深度图，分辨率 640x576。

    :param merged_vertices: 顶点列表 [[x, y, z], ...]
    :param merged_faces: 面列表 [[count, v1, v2, v3], ...]
    :param camera_points: 相机位置 [x, y, z]
    :param init_place_points: 初始放置点位置 [x, y, z]
    :return: NumPy array of shape (576, 640)
    """
    vertices_np = np.array([v[:3] for v in merged_vertices], dtype=np.float32)
    faces_np = np.hstack(merged_faces).astype(np.int32)  # PyVista expects faces in a specific format

    # 创建 mesh
    mesh = pv.PolyData(vertices_np, faces_np)

    # 构造相机矩阵：朝向 place 点
    camera_pos = np.array(camera_points, dtype=np.float32)
    place_pos = np.array(init_place_points, dtype=np.float32)
    place_direction = place_pos.copy()
    place_direction[2] = 0.1  # 投影到 XY 平面

    # 初始化PyVista绘图窗口（对于无头环境，请确保设置off_screen=True）
    plotter = pv.Plotter(off_screen=True, window_size=[640, 576])

    # 添加 mesh 到场景
    plotter.add_mesh(mesh, color=[0.3, 0.3, 0.3])

    # 设置相机
    plotter.camera.position = camera_pos
    plotter.camera.focal_point = place_direction
    plotter.camera.up = [0, 0, 1]  # 默认上方向为Z轴正方向

    # 渲染并获取深度图像
    plotter.show(auto_close=False)
    depth_image = plotter.get_zbuffer()

    # 关闭绘图器释放资源
    plotter.close()

    return depth_image