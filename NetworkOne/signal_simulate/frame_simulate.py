import numpy as np
import cupy as cp
import time
from scipy.signal import gausspulse

# -----------------------------
# 雷达参数配置
# -----------------------------
fb = 1.4e9
fc = 7.29e9
bw = fb / fc
bwr = -10

sample_rate = 23.328e9
light_speed = 2.998e8
decimate_factor = 8
real_sample_rate = sample_rate / decimate_factor
sample_interval = 1.0 / real_sample_rate

# 设置采样点数
start_bin = 0
end_bin = 96
t = cp.arange(start_bin * sample_interval, end_bin * sample_interval, sample_interval)
sample_point = t.size

# 信号载波生成
cos_carrier = lambda t: cp.cos(2 * cp.pi * fc * t)
sin_carrier = lambda t: cp.sin(2 * cp.pi * fc * t)

# 使用高斯脉冲作为发射信号
g = lambda t: gausspulse(t, fc=fc, bw=bw, bwr=bwr, retquad=True, retenv=True)[2]

# -----------------------------
# 物体运动参数
# -----------------------------
FPS = 40  # 帧率 (Hz)
moving_speed = 0.01  # 移动速度 (m/s)
moving_step = moving_speed / FPS  # 每步移动距离

# 发射功率和接收增益
tx_power_dbm = 6.3
rx_gain_db = 14.1

# 将dBm和dB转换为线性值
tx_power = 10 ** (tx_power_dbm / 10) / 1000  # 转换为瓦特
rx_gain = 10 ** (rx_gain_db / 10)
sigma = 0.285  # 高斯衰减的标准差


def signalCal(camera, valid_faces, valid_centers, valid_normals, valid_vertices, vertex_intensity):
    receive_signal = cp.zeros(sample_point, dtype=cp.complex128)
    # 计算面面积
    valid_edge1 = valid_vertices[:, 1] - valid_vertices[:, 0]  # (num_valid_faces, 3)
    valid_edge2 = valid_vertices[:, 2] - valid_vertices[:, 0]  # (num_valid_faces, 3)
    valid_areas = 0.5 * cp.linalg.norm(cp.cross(valid_edge1, valid_edge2), axis=1)  # (num_valid_faces,)

    # 计算入射向量与距离
    Inner = valid_centers - camera  # (num_valid_faces, 3)
    distance_I = cp.linalg.norm(Inner, axis=1)  # (num_valid_faces,)

    # 归一化入射向量
    I_normalized = Inner / distance_I[:, cp.newaxis]  # (num_valid_faces, 3)

    # 计算理想出射向量（镜面反射）
    dot_products = cp.sum(I_normalized * valid_normals, axis=1, keepdims=True)  # (num_valid_faces, 1)
    R = I_normalized - 2 * dot_products * valid_normals  # (num_valid_faces, 3)

    # 计算实际出射向量（面中心到雷达）
    cl = camera - valid_centers  # (num_valid_faces, 3)
    distance_cl = cp.linalg.norm(cl, axis=1)  # (num_valid_faces,)

    # 归一化实际出射向量
    cl_normalized = cl / distance_cl[:, cp.newaxis]  # (num_valid_faces, 3)

    # 计算反射角度偏差
    dot_products = cp.sum(cl_normalized * R, axis=1)  # (num_valid_faces,)
    dot_products = cp.clip(dot_products, -1.0, 1.0)
    gamma = cp.arccos(dot_products)  # 角度（弧度）

    # 飞行时间（往返距离/光速）
    ToF = 2 * distance_I / light_speed  # (num_valid_faces,)

    # 面的反射强度（三个顶点的平均值）
    vertex_idx_0 = valid_faces[:, 0]
    vertex_idx_1 = valid_faces[:, 1]
    vertex_idx_2 = valid_faces[:, 2]
    intensity = (vertex_intensity[vertex_idx_0] +
                 vertex_intensity[vertex_idx_1] +
                 vertex_intensity[vertex_idx_2]) / 3.0  # (num_valid_faces,)

    # 入射向量与法向量夹角的余弦值
    cos_incident_angle = cp.clip(cp.abs(cp.sum(I_normalized * valid_normals, axis=1)), 0.0, 1.0)
    # 实际出射向量与法向量夹角的余弦值
    cos_cl_angle = cp.clip(cp.sum(cl_normalized * valid_normals, axis=1), -1.0, 1.0)

    # 计算信号幅值与相位
    distance_att = valid_areas / (distance_I ** 2)  # 距离衰减因子
    angle_att = cp.exp(- (gamma ** 2) / (2 * sigma ** 2))  # 角度衰减因子
    # 0.5 为下变频引入的系数
    base_amplitude = 0.5 * distance_att * angle_att * intensity * cos_incident_angle * cos_cl_angle  # 基础幅值

    # 考虑发射功率和接收增益
    base_amplitude = base_amplitude * cp.sqrt(tx_power * rx_gain)

    # -----------------------------
    # 信号叠加
    # -----------------------------
    # 时间向量 (sample_point,)
    t_vector = t  # 已为cp.ndarray

    # 计算时间差矩阵 (sample_point, num_valid_faces)
    time_diffs = t_vector[:, cp.newaxis] - ToF[cp.newaxis, :]

    # 计算高斯脉冲调制矩阵 (sample_point, num_valid_faces)
    g_matrix = g(time_diffs)

    # 计算信号幅值矩阵与相位矩阵
    A_matrix = base_amplitude[cp.newaxis, :] * g_matrix  # 幅值矩阵
    theta_matrix = 2 * cp.pi * fc * ToF[cp.newaxis, :]  # 相位矩阵

    # 计算复数信号并叠加
    complex_signal = A_matrix * cp.exp(-1j * theta_matrix)  # 复数信号矩阵
    receive_signal = cp.sum(complex_signal, axis=1)  # 叠加

    return receive_signal


def frame_simulate(merged_vertices: cp.ndarray, merged_faces: cp.ndarray, mask_vertices: cp.ndarray,
                   mask_faces: cp.ndarray, camera_points: cp.ndarray,
                   init_place_points: cp.ndarray) -> cp.ndarray:
    """
    模拟某一帧的 UWB 雷达回波信号（带进度输出的向量化优化版本）
    :param merged_vertices: 顶点数组，shape (N, 10)，格式[x, y, z, r, g, b, nx, ny, nz, i]
    :param merged_faces: 面数组，shape (M, 4)，格式[count, idx1, idx2, idx3]
    :param mask_vertices: 顶点数组，shape (N, 10)，格式[x, y, z, r, g, b, nx, ny, nz, i]
    :param mask_faces: 面数组，shape (M, 4)，格式[count, idx1, idx2, idx3]
    :param camera_points: 相机位置数组，shape (3,)
    :param init_place_points: 初始放置点位置数组，shape (3,)
    """
    start_time = time.time()
    print("===== 开始计算雷达回波信号 =====")
    print(f"总采样点数: {sample_point} | 顶点数: {merged_vertices.shape[0]} | 面数: {merged_faces.shape[0]}")

    # 初始化接收信号（复数值数组）
    receive_signal = cp.zeros(sample_point, dtype=cp.complex128)

    # 相机和放置点坐标（已为cp.ndarray）
    camera = camera_points  # 雷达位置 [x,y,z]
    place = init_place_points  # 参考点位置 [x,y,z]

    # -----------------------------
    # 解析顶点数据（提取坐标和属性）
    # -----------------------------
    vertices = merged_vertices[:, :3]  # 提取前3列作为坐标 (N, 3)
    vertices_mask = mask_vertices[:, :3]  # 提取前3列作为坐标 (N, 3)

    # -----------------------------
    # 解析面数据（三角形索引）
    # -----------------------------
    faces = merged_faces[:, 1:4]  # 提取后3列作为顶点索引 (M, 3)
    faces_mask = mask_faces[:, 1:4]  # 提取后3列作为顶点索引 (M, 3)

    # -----------------------------
    # 朝向确定（空间分割）
    # -----------------------------
    # 计算从雷达出发到place的向量
    radar_to_place = place - camera  # 向量方向：雷达 -> place
    # 将向量投影到xy平面（z分量设为0）
    radar_to_place[2] = 0.0  # 直接置零z分量

    radar_to_place_norm = cp.linalg.norm(radar_to_place)
    if radar_to_place_norm < 1e-6:  # 避免除以零
        radar_to_place_norm = 1e-6
    radar_to_place_normalized = radar_to_place / radar_to_place_norm  # 归一化

    # 计算每个面的中心点
    face_centers = cp.mean(vertices[faces], axis=1)  # (M, 3)

    # 计算从雷达出发到面中心点的向量
    center_to_camera = face_centers - camera  # (M, 3)

    # 计算两个向量的点积（保留点积≥0的面）
    dot_products = cp.sum(center_to_camera * radar_to_place_normalized, axis=1)  # (M,)
    valid_orient_mask = dot_products >= 0

    # -----------------------------
    # 背面剔除
    # -----------------------------
    # 计算每个三角形面的法向量
    v0 = vertices[faces[:, 0]]  # (M, 3)
    v1 = vertices[faces[:, 1]]  # (M, 3)
    v2 = vertices[faces[:, 2]]  # (M, 3)
    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0  # (M, 3)
    face_normals = cp.cross(edge1, edge2)  # (M, 3)

    # 归一化法向量
    face_normal_norms = cp.linalg.norm(face_normals, axis=1, keepdims=True)  # (M, 1)
    face_normal_norms[face_normal_norms < 1e-6] = 1e-6
    face_normals = face_normals / face_normal_norms  # (M, 3)

    # 计算视线向量与法向量的夹角余弦值
    view_vectors = face_centers - camera  # (M, 3)
    view_norms = cp.linalg.norm(view_vectors, axis=1, keepdims=True)  # (M, 1)
    view_norms[view_norms < 1e-6] = 1e-6
    view_vectors_normalized = view_vectors / view_norms  # (M, 3)
    cos_angles = cp.sum(view_vectors_normalized * face_normals, axis=1)  # (M,)

    # 保留可见面
    valid_visible_mask = cos_angles < 0  # (M,)
    valid_faces_mask = valid_orient_mask & valid_visible_mask  # (M,)
    valid_face_indices = cp.where(valid_faces_mask)[0]  # 有效面的索引
    valid_total_count = valid_face_indices.size

    # 若没有有效面，返回零信号
    if valid_total_count == 0:
        print("警告：物品与环境组合体无有效面参与信号计算，返回零信号")
        return receive_signal
    else:
        print(f"物品有效面数目：{valid_total_count}")
    # -----------------------------
    # 对掩体背面剔除
    # -----------------------------
    mask_v0 = vertices_mask[faces_mask[:, 0]]  # (M, 3)
    mask_v1 = vertices_mask[faces_mask[:, 1]]  # (M, 3)
    mask_v2 = vertices_mask[faces_mask[:, 2]]  # (M, 3)
    mask_edge1 = mask_v1 - mask_v0  # (M, 3)
    mask_edge2 = mask_v2 - mask_v0  # (M, 3)
    mask_face_normals = cp.cross(mask_edge1, mask_edge2)  # (M, 3)

    # 归一化法向量
    mask_face_normal_norms = cp.linalg.norm(mask_face_normals, axis=1, keepdims=True)  # (M, 1)
    mask_face_normal_norms[mask_face_normal_norms < 1e-6] = 1e-6
    mask_face_normals = mask_face_normals / mask_face_normal_norms  # (M, 3)

    # 计算视线向量（雷达到面中心）
    mask_face_centers = cp.mean(vertices_mask[faces_mask], axis=1)  # (M, 3)
    mask_view_vectors = mask_face_centers - camera  # (M, 3)
    mask_view_norms = cp.linalg.norm(mask_view_vectors, axis=1, keepdims=True)  # (M, 1)
    mask_view_norms[mask_view_norms < 1e-6] = 1e-6
    mask_view_vectors_normalized = mask_view_vectors / mask_view_norms  # (M, 3)

    # 计算视线向量与法向量的夹角余弦值
    mask_cos_angles = cp.sum(mask_view_vectors_normalized * mask_face_normals, axis=1)  # (M,)

    # 保留可见面
    mask_valid_visible_mask = mask_cos_angles < 0  # (M,)

    # 有效面索引
    mask_valid_face_indices = cp.where(mask_valid_visible_mask)[0]  # 有效面的索引
    mask_valid_total_count = mask_valid_face_indices.size

    # 若没有有效面，返回零信号
    if mask_valid_total_count == 0:
        print("警告：掩体无有效面参与信号计算，正在保存mask数据到PLY文件...")
        return receive_signal
    else:
        print(f"掩体有效面数目：{mask_valid_total_count}")
    # -----------------------------
    # 对物品和掩体合并
    # -----------------------------
    # -----------------------------
    # 提取物品有效面及顶点信息
    # -----------------------------
    # 1. 提取物品有效面（原始面信息）
    item_valid_faces = merged_faces[valid_face_indices]  # shape (V, 4)，V为物品有效面数量
    item_faces_adj = item_valid_faces.copy()
    # 2. 提取有效面涉及的所有顶点索引（去重）
    item_vertex_ids = item_valid_faces[:, 1:4].flatten()  # 展平所有顶点索引
    item_unique_vertex_ids = cp.unique(item_vertex_ids)
    item_valid_vertices = merged_vertices[item_unique_vertex_ids]
    # 构建原始索引→新索引的映射数组（GPU端）
    max_old_id = int(item_unique_vertex_ids.max())
    item_id_map_arr = cp.full(max_old_id + 1, -1, dtype=cp.int32)
    item_id_map_arr[item_unique_vertex_ids] = cp.arange(len(item_unique_vertex_ids))
    # 批量映射索引（GPU端并行）
    item_faces_adj[:, 1:4] = item_id_map_arr[item_faces_adj[:, 1:4]]

    # -----------------------------
    # 提取掩体有效面及顶点信息
    # -----------------------------
    # 1. 提取掩体有效面（原始面信息）
    mask_valid_faces = mask_faces[mask_valid_face_indices]  # shape (W, 4)，W为掩体有效面数量
    mask_faces_adj = mask_valid_faces.copy()  # 初始化调整后的面数组

    # 2. 提取有效面涉及的所有顶点索引（去重）
    mask_vertex_ids = mask_valid_faces[:, 1:4].flatten()  # 展平所有顶点索引（shape: (3W,)）
    mask_unique_vertex_ids = cp.unique(mask_vertex_ids)  # 去重后的顶点索引（shape: (U2,)）

    # 3. 提取对应的顶点数据
    mask_valid_vertices = mask_vertices[mask_unique_vertex_ids]  # shape (U2, 10)，U2为掩体有效顶点数量

    # 4. 建立顶点索引映射（原始索引 -> 合并后新索引，新索引偏移物品顶点数量）

    # 构建GPU端映射数组（原始索引 -> 新索引）
    max_mask_old_id = int(mask_unique_vertex_ids.max())  # 最大原始索引
    mask_id_map_arr = cp.full(max_mask_old_id + 1, -1, dtype=cp.int32)  # 初始化为-1（无效值）
    # 新索引 = 掩体顶点自身索引（0~U2-1） + 物品顶点数量（确保合并后索引连续）
    mask_id_map_arr[mask_unique_vertex_ids] = cp.arange(len(mask_unique_vertex_ids)) + len(item_unique_vertex_ids)
    # 5. 批量映射索引（GPU端并行，替换CPU循环）
    mask_faces_adj[:, 1:4] = mask_id_map_arr[mask_faces_adj[:, 1:4]]  # 直接用数组索引完成批量转换

    # -----------------------------
    # 合并物品和掩体的顶点与面数组
    # -----------------------------
    # 合并顶点数组（物品顶点在前，掩体顶点在后）
    combined_vertices = cp.vstack((item_valid_vertices, mask_valid_vertices))  # shape (U1+U2, 10)

    # 合并面数组（物品面在前，掩体面在后）
    combined_faces = cp.vstack((item_faces_adj, mask_faces_adj))  # shape (V+W, 4)
    print(f"筛除背面后：顶点数: {combined_vertices.shape[0]} | 面数: {combined_faces.shape[0]}")
    # -----------------------------
    # 信号计算与叠加
    # -----------------------------
    # 信号参数
    combined_valid_vertices = combined_vertices[:, :3]  # 提取前3列作为坐标 (N, 3)
    combined_valid_vertex_intensity = combined_vertices[:, -1]  # 提取最后1列作为强度 (N,)
    combined_valid_faces = combined_faces[:, 1:4]  # 提取后3列作为顶点索引 (M, 3)

    combined_face_vertices = combined_valid_vertices[combined_valid_faces]  # (M, 3, 3)

    # 计算每个三角形面的法向量
    combined_valid_v0 = combined_valid_vertices[combined_valid_faces[:, 0]]  # (M, 3)
    combined_valid_v1 = combined_valid_vertices[combined_valid_faces[:, 1]]  # (M, 3)
    combined_valid_v2 = combined_valid_vertices[combined_valid_faces[:, 2]]  # (M, 3)
    combined_valid_edge1 = combined_valid_v1 - combined_valid_v0  # (M, 3)
    combined_valid_edge2 = combined_valid_v2 - combined_valid_v0  # (M, 3)
    combined_valid_face_normals = cp.cross(combined_valid_edge1, combined_valid_edge2)  # (M, 3)

    # 归一化法向量
    combined_valid_face_normal_norms = cp.linalg.norm(combined_valid_face_normals, axis=1, keepdims=True)  # (M, 1)
    combined_valid_face_normal_norms[combined_valid_face_normal_norms < 1e-6] = 1e-6
    combined_valid_face_normals = combined_valid_face_normals / combined_valid_face_normal_norms  # (M, 3)

    combined_valid_face_centers = cp.mean(combined_valid_vertices[combined_valid_faces], axis=1)  # (M, 3)
    receive_signal = signalCal(camera, combined_valid_faces, combined_valid_face_centers, combined_valid_face_normals,
                               combined_face_vertices, combined_valid_vertex_intensity)
    # 总耗时
    total_time = time.time() - start_time
    print(f"\n===== 计算完成，总耗时：{total_time:.2f}秒 =====")
    return receive_signal
