import numpy as np
import os
import scipy.io as sio
from PIL import Image


def save_numpy_array_to_mat(numpy_array, filename):
    """
    将 NumPy 数组保存为 .mat 文件
    :param numpy_array: 要保存的 NumPy 数组
    :param filename: 保存路径（需以 .mat 结尾）
    """
    # 确保文件名以 .mat 结尾
    if not filename.endswith('.mat'):
        filename += '.mat'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 保存为 .mat 文件
    sio.savemat(filename, {'data': numpy_array})


def save_numpy_depths_to_pngs(depth_array, output_dir, file_name, max_depth=None):
    """
    将float32格式的深度数组保存为16位PNG深度图
    使用 depth_array / max_depth * 65535 的归一化方式

    参数:
        depth_array: 输入的float32格式深度数组（np.array）
        output_dir: 输出目录
        file_name: 输出文件名（不含扩展名）
        max_depth: 可选参数，指定最大深度值。若为None，则使用非零值的最大值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 验证输入数组格式
    if depth_array.dtype != np.float32:
        depth_array = depth_array.astype(np.float32)

    # 处理空数组
    if depth_array.size == 0:
        print(f"警告: 输入的深度数组为空，将保存全零深度图")
        depth_16bit = np.zeros((1, 1), dtype=np.uint16)
    else:
        # 确定归一化的最大深度值
        if max_depth is None:
            valid_depths = depth_array[depth_array > 0]
            if valid_depths.size > 0:
                max_depth = np.max(valid_depths)
            else:
                max_depth = 1.0  # 默认值，防止全零数组出错

        # 归一化方式: depth_array / max_depth * 65535
        if max_depth > 0:
            depth_16bit = (depth_array / max_depth * 65535).astype(np.uint16)
            # 限制超出范围的值
            depth_16bit = np.clip(depth_16bit, 0, 65535)
        else:
            depth_16bit = np.zeros_like(depth_array, dtype=np.uint16)

    # 保存为16位PNG
    output_path = os.path.join(output_dir, f"{file_name}.png")
    try:
        # 使用PIL库保存16位PNG
        img = Image.fromarray(depth_16bit)
        img.save(output_path, "PNG")
        print(f"成功保存深度图到 {output_path}")
        return True
    except Exception as e:
        print(f"保存深度图失败: {str(e)}")
        return False


def downsample_faces(faces: np.ndarray, num_faces: int) -> np.ndarray:
    num_total_faces = len(faces)
    if num_total_faces <= num_faces:
        return faces
    # 随机采样面索引
    sampled_face_indices = np.random.choice(num_total_faces, num_faces, replace=False)
    sampled_faces = faces[sampled_face_indices]
    return sampled_faces


def update_vertices_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 提取有效顶点索引（所有面中出现的顶点）
    valid_vertex_indices = np.unique(faces[:, 1:])  # 面的第一个元素是顶点数，后面是索引
    # 创建索引映射（原索引 -> 新索引）
    index_mapping = {idx: i for i, idx in enumerate(valid_vertex_indices)}
    # 筛选有效顶点
    valid_vertices = vertices[valid_vertex_indices]
    # 调整面的顶点索引
    valid_faces = []
    for face in faces:
        vertex_count = face[0]
        vertex_indices = [index_mapping[idx] for idx in face[1:]]
        adjusted_face = [vertex_count] + vertex_indices
        valid_faces.append(adjusted_face)
    return valid_vertices, np.array(valid_faces)


def compute_aabb_center(vertices: np.ndarray) -> np.ndarray:
    """计算顶点集合的AABB下底面中心点"""
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    min_z = np.min(vertices[:, 2])
    return np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, min_z])


def transform_vertices(vertices: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """对顶点应用平移变换"""
    transformed = vertices.copy()
    # 平移坐标（前3列）
    transformed[:, :3] += translation[:3]
    return transformed


def process_scene_intensity(scene_vertices: np.ndarray) -> np.ndarray:
    """处理场景顶点的intensity属性，进行归一化"""
    # 提取intensity属性（最后一列）
    intensity_values = scene_vertices[:, -1].copy()

    # 确定intensity上下限（99.7%分位数）
    lower_bound = np.percentile(intensity_values, 0.0)
    upper_bound = np.percentile(intensity_values, 99.7)

    # 截断超出范围的值
    intensity_values = np.clip(intensity_values, lower_bound, upper_bound)

    # 归一化到0-1之间
    min_intensity = np.min(intensity_values)
    max_intensity = np.max(intensity_values)
    if max_intensity != min_intensity:
        intensity_values = (intensity_values - min_intensity) / (max_intensity - min_intensity)
    else:
        intensity_values = np.zeros_like(intensity_values)

    # 更新intensity属性
    processed = scene_vertices.copy()
    processed[:, -1] = intensity_values
    return processed


def merge_meshes(scene_vertices: np.ndarray, scene_faces: np.ndarray,
                 item_vertices: np.ndarray, item_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """合并场景和物品的顶点和面"""
    # 合并顶点
    merged_vertices = np.vstack((scene_vertices, item_vertices))
    scene_vertex_count = len(scene_vertices)

    # 调整物品面的顶点索引（加上场景顶点数量的偏移）
    adjusted_item_faces = item_faces.copy()
    adjusted_item_faces[:, 1:] += scene_vertex_count  # 面的第一个元素是顶点数，后面是索引

    # 合并面
    merged_faces = np.vstack((scene_faces, adjusted_item_faces))
    return merged_vertices, merged_faces


def place_mesh(place_points: np.ndarray, scene_vertices: np.ndarray, scene_faces: np.ndarray,
               item_vertices: np.ndarray, item_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将物品网格放置到场景中指定的位置"""
    # 处理场景顶点的intensity属性
    processed_scene_vertices = process_scene_intensity(scene_vertices)

    # 计算物品AABB下底面中心点
    aabb_center = compute_aabb_center(item_vertices)

    # 计算平移向量
    translation = place_points - aabb_center

    # 应用平移变换到物品顶点
    transformed_item_vertices = transform_vertices(item_vertices, translation)

    # 合并网格
    merged_vertices, merged_faces = merge_meshes(
        processed_scene_vertices, scene_faces,
        transformed_item_vertices, item_faces
    )

    return merged_vertices, merged_faces


def filter_faces_by_distance(merged_vertices: np.ndarray, merged_faces: np.ndarray,
                             camera_point: np.ndarray, distance_threshold=6.0) -> tuple[np.ndarray, np.ndarray]:
    # 提取顶点坐标（前3列）
    vertices = merged_vertices[:, :3]

    # 计算每个面的中心点
    face_indices = merged_faces[:, 1:]  # 面的顶点索引（排除第一个元素）
    face_centers = np.mean(vertices[face_indices], axis=1)

    # 计算中心点与相机的距离
    distances = np.linalg.norm(face_centers - camera_point, axis=1)

    # 筛选距离小于阈值的面
    valid_mask = distances < distance_threshold
    valid_faces = merged_faces[valid_mask]

    return merged_vertices, valid_faces