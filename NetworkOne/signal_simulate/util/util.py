import cupy as cp
import os

import numpy as np
import scipy.io as sio


def save_cupy_array_to_mat(cupy_array, filename):
    """
    将 CuPy 数组保存为 .mat 文件
    :param cupy_array: 要保存的 CuPy 数组
    :param filename: 保存路径（需以 .mat 结尾）
    """
    # 确保文件名以 .mat 结尾
    if not filename.endswith('.mat'):
        filename += '.mat'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 将 CuPy 数组转换为 NumPy 数组
    numpy_array = cp.asnumpy(cupy_array)

    # 保存为 .mat 文件
    sio.savemat(filename, {'data': numpy_array})

def save_cupy_array_to_npy(cupy_array, filename):
    """
    将 CuPy 数组保存为 .npy 文件
    :param cupy_array: 要保存的 CuPy 数组
    :param filename: 保存路径（需以 .npy 结尾）
    """
    # 确保文件名以 .npy 结尾
    if not filename.endswith('.npy'):
        filename += '.npy'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 将 CuPy 数组转换为 NumPy 数组
    numpy_array = cp.asnumpy(cupy_array)

    # 保存为 .npy 文件（numpy.save 直接保存数组，无需包装字典）
    np.save(filename, numpy_array)

def downsample_faces(faces: cp.ndarray, num_faces: int) -> cp.ndarray:
    """
    从 faces 中随机采样指定数量的面
    :param faces: 面数据，形状为 [N, face_info]
    :param num_faces: 需要采样的面数量
    :return: 采样后的面数据
    """
    num_total_faces = faces.shape[0]
    if num_total_faces <= num_faces:
        return faces

    sampled_face_indices = cp.random.choice(num_total_faces, num_faces, replace=False)
    sampled_faces = faces[sampled_face_indices]
    return sampled_faces


def update_vertices_faces(vertices: cp.ndarray, faces: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    根据 faces 过滤 vertices，并更新 faces 中的顶点索引
    :param vertices: 顶点数据，形状为 [N, 3]
    :param faces: 面数据，形状为 [M, face_info]
    :return: 过滤后的顶点数据和更新后的面数据
    """
    # 提取所有使用的顶点索引
    valid_vertex_indices = cp.unique(faces[:, 1:])

    # 创建索引映射
    new_indices = cp.zeros(vertices.shape[0], dtype=cp.int32)
    new_indices[valid_vertex_indices] = cp.arange(len(valid_vertex_indices))

    # 过滤顶点
    valid_vertices = vertices[valid_vertex_indices]

    # 更新面中的顶点索引
    valid_faces = faces.copy()
    valid_faces[:, 1:] = new_indices[faces[:, 1:]]

    return valid_vertices, valid_faces


def compute_aabb_center(vertices: cp.ndarray) -> cp.ndarray:
    """
    计算顶点集合的AABB下底面中心点
    :param vertices: 顶点数据，形状为 [N, 3]
    :return: AABB下底面中心点，形状为 [3]
    """
    min_x = cp.min(vertices[:, 0])
    max_x = cp.max(vertices[:, 0])
    min_y = cp.min(vertices[:, 1])
    max_y = cp.max(vertices[:, 1])
    min_z = cp.min(vertices[:, 2])

    return cp.array([(min_x + max_x) / 2, (min_y + max_y) / 2, min_z])


def transform_vertices(vertices: cp.ndarray, translation: cp.ndarray) -> cp.ndarray:
    """
    对顶点应用平移变换
    :param vertices: 顶点数据，形状为 [N, D]
    :param translation: 平移向量，形状为 [3]
    :return: 平移后的顶点数据，形状为 [N, D]
    """
    # 创建平移矩阵
    translation_matrix = cp.zeros((vertices.shape[0], vertices.shape[1]))
    translation_matrix[:, :3] = translation

    # 应用平移
    transformed = vertices + translation_matrix

    return transformed


def process_scene_intensity(scene_vertices: cp.ndarray) -> cp.ndarray:
    """
    处理场景顶点的intensity属性，进行归一化
    :param scene_vertices: 场景顶点数据，形状为 [N, D]，最后一列为intensity
    :return: 处理后的顶点数据，形状为 [N, D]
    """
    intensity_values = scene_vertices[:, -1]

    # 确定intensity上限和下限，使得intensity在上下限之间的点数占全部点数的99％
    lower_bound = cp.percentile(intensity_values, 0.0)
    upper_bound = cp.percentile(intensity_values, 99.7)

    # 将低于下限的点的intensity属性置为下限，高于上限的点的intensity属性置为上限
    intensity_values = cp.clip(intensity_values, lower_bound, upper_bound)

    # 将所有点的intensity属性归一化到0-1之间
    min_intensity = cp.min(intensity_values)
    max_intensity = cp.max(intensity_values)

    if max_intensity != min_intensity:
        intensity_values = (intensity_values - min_intensity) / (max_intensity - min_intensity)
    else:
        intensity_values = cp.zeros_like(intensity_values)

    # 更新scene_vertices中的intensity属性
    processed_vertices = scene_vertices.copy()
    processed_vertices[:, -1] = intensity_values

    return processed_vertices


def merge_meshes(scene_vertices: cp.ndarray, scene_faces: cp.ndarray, item_vertices: cp.ndarray,
                 item_faces: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    合并场景和物品的顶点和面
    :param scene_vertices: 场景顶点数据，形状为 [N1, D]
    :param scene_faces: 场景面数据，形状为 [M1, face_info]
    :param item_vertices: 物品顶点数据，形状为 [N2, D]
    :param item_faces: 物品面数据，形状为 [M2, face_info]
    :return: 合并后的顶点数据和面数据
    """
    # 合并顶点
    merged_vertices = cp.vstack([scene_vertices, item_vertices])

    # 更新物品面的顶点索引
    scene_vertex_count = scene_vertices.shape[0]
    item_faces = item_faces.copy()
    item_faces[:, 1:] += scene_vertex_count

    # 合并面
    merged_faces = cp.vstack([scene_faces, item_faces])

    return merged_vertices, merged_faces


def place_mesh(place_points: cp.ndarray, scene_vertices: cp.ndarray, scene_faces: cp.ndarray,
               item_vertices: cp.ndarray, item_faces: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    将物品网格放置到场景中指定的位置
    :param place_points: 放置点，形状为 [3]
    :param scene_vertices: 场景顶点数据，形状为 [N1, D]
    :param scene_faces: 场景面数据，形状为 [M1, face_info]
    :param item_vertices: 物品顶点数据，形状为 [N2, D]
    :param item_faces: 物品面数据，形状为 [M2, face_info]
    :return: 合并后的顶点数据和面数据
    """
    # 处理场景顶点的intensity属性
    processed_scene_vertices = process_scene_intensity(scene_vertices)

    # 计算AABB下底面中心点
    aabb_center = compute_aabb_center(item_vertices)

    # 计算平移向量
    translation = place_points - aabb_center

    # 应用平移变换到物品顶点
    transformed_item_vertices = transform_vertices(item_vertices, translation)

    # 合并网格
    merged_vertices, merged_faces = merge_meshes(
        processed_scene_vertices,
        scene_faces,
        transformed_item_vertices,
        item_faces
    )

    return merged_vertices, merged_faces


def filter_faces_by_distance(merged_vertices: cp.ndarray, merged_faces: cp.ndarray,
                             camera_point: cp.ndarray, distance_threshold=6.0) -> tuple[cp.ndarray, cp.ndarray]:
    """
    基于距离过滤面
    :param merged_vertices: 合并后的顶点数据，形状为 [N, D]
    :param merged_faces: 合并后的面数据，形状为 [M, face_info]
    :param camera_point: 相机位置，形状为 [3]
    :param distance_threshold: 距离阈值
    :return: 过滤后的顶点数据和面数据
    """
    # 提取顶点坐标
    vertices = merged_vertices[:, :3]

    # 计算每个面的中心点
    face_centers = cp.zeros((merged_faces.shape[0], 3))

    # 处理不同顶点数的面
    for i in range(merged_faces.shape[0]):
        vertex_count = merged_faces[i, 0]
        vertex_indices = merged_faces[i, 1:vertex_count + 1]
        face_centers[i] = cp.mean(vertices[vertex_indices], axis=0)

    # 计算中心点与相机的距离
    distances = cp.linalg.norm(face_centers - camera_point, axis=1)

    # 只保留距离小于阈值的面
    valid_face_mask = distances < distance_threshold
    valid_faces = merged_faces[valid_face_mask]

    # 更新顶点和面
    valid_vertices, valid_faces = update_vertices_faces(merged_vertices, valid_faces)

    return valid_vertices, valid_faces