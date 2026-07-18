from typing import Any

import cupy as cp
import os

from cupy import ndarray
from numpy import dtype

from frame_simulate import sample_point, frame_simulate
from util.read_file import read_place_txt, parse_scene_ply, parse_item_ply, parse_mask_ply
from util.util import place_mesh, downsample_faces, update_vertices_faces, \
    filter_faces_by_distance, save_cupy_array_to_npy

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')


def object_move_signal_simulate(
        item_vertices: cp.ndarray,  # shape (N, 10)
        item_faces: cp.ndarray,  # shape (M, 4)
        scene_vertices: cp.ndarray,  # shape (P, 10)
        scene_faces: cp.ndarray,  # shape (Q, 4)
        mask_vertices: cp.ndarray,  # shape (P, 10)
        mask_faces: cp.ndarray,  # shape (Q, 4)
        camera_points: cp.ndarray,  # shape (3,)
        init_place_points: cp.ndarray,  # shape (3,)
        start_path_points: cp.ndarray,  # shape (3,)
        end_path_points: cp.ndarray,  # shape (3,)
        steps: int,
) -> ndarray[Any, dtype[Any]]:
    """模拟物体在多个移动步长下的雷达信号变化（优化内存版）"""
    # 初始化信号数组（仅保留必要结果数组）
    signal = cp.zeros((steps, sample_point), dtype=cp.complex128)

    # 预处理掩体顶点数据（提取x,y,z坐标并计算底部中心）
    x_coords = mask_vertices[:, 0]
    y_coords = mask_vertices[:, 1]
    z_coords = mask_vertices[:, 2]
    min_x, max_x = cp.min(x_coords), cp.max(x_coords)
    min_y, max_y = cp.min(y_coords), cp.max(y_coords)
    min_z = cp.min(z_coords)
    mask_botton_center = cp.array([(min_x + max_x) / 2, (min_y + max_y) / 2, min_z])

    # 释放预处理坐标变量（已无用）
    del x_coords, y_coords, z_coords, min_x, max_x, min_y, max_y, min_z
    cp.get_default_memory_pool().free_all_blocks()
    # 计算所有移动位置（线性插值生成路径点）
    place_points_list = cp.linspace(start_path_points, end_path_points, steps)  # shape (steps, 3)

    # 预处理场景：过滤远距离面并更新顶点索引
    distance_threshold = 6.0
    rest_vertices, rest_faces = filter_faces_by_distance(
        scene_vertices, scene_faces, camera_points, distance_threshold
    )
    logging.info(f"filter completed")
    # 释放原始场景顶点/面（预处理后已无用）
    del scene_vertices, scene_faces
    cp.get_default_memory_pool().free_all_blocks()

    rest_vertices, rest_faces = update_vertices_faces(rest_vertices, rest_faces)
    logging.info(f"update completed")
    logging.info(f"Starting parallel processing for {steps} steps")

    # 循环处理每个步长（动态计算偏移，避免预分配大数组）
    for i in range(steps):
        # 1. 动态计算当前步长的物品顶点偏移（仅保留当前步数据）
        mask_translation = place_points_list[i] - mask_botton_center

        mask_vertices_offset = mask_vertices.copy()
        mask_vertices_offset[:, :3] += mask_translation

        # 2. 合并场景与当前位置的物品
        place_points = place_points_list[i]
        merged_vertices, merged_faces = place_mesh(
            place_points, rest_vertices, rest_faces, item_vertices, item_faces
        )
        logging.info(f"place completed (step {i + 1}/{steps})")

        # 3. 计算单物品信号和合并信号
        signal[i] = frame_simulate(
            merged_vertices, merged_faces, mask_vertices_offset, mask_faces, camera_points, init_place_points
        )

        # 4. 关键：清理当前迭代的中间变量（立即释放内存）
        del mask_translation, place_points, merged_vertices, merged_faces, mask_vertices_offset
        # 强制释放内存池未使用块（避免缓存累积）
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # 释放预处理场景变量（循环结束后已无用）
    del rest_vertices, rest_faces, place_points_list
    cp.get_default_memory_pool().free_all_blocks()

    logging.info(f"All {steps} steps completed")
    return signal


def both_object_move_signal_simulate(txt_file, scene_file, item_file, mask_file, out_dir):
    # 读取场景和物品数据
    scene_vertices_list, scene_faces_list = parse_scene_ply(scene_file)
    scene_vertices = cp.array(scene_vertices_list, dtype=cp.float32)
    scene_faces = cp.array(scene_faces_list, dtype=cp.int32)

    item_vertices_list, item_faces_list = parse_item_ply(item_file)
    item_vertices = cp.array(item_vertices_list, dtype=cp.float32)
    item_faces = cp.array(item_faces_list, dtype=cp.int32)

    mask_vertices_list, mask_faces_list = parse_mask_ply(mask_file)
    mask_vertices = cp.array(mask_vertices_list, dtype=cp.float32)
    mask_faces = cp.array(mask_faces_list, dtype=cp.int32)

    del scene_vertices_list, scene_faces_list, item_vertices_list, item_faces_list, mask_vertices_list, mask_faces_list
    cp.get_default_memory_pool().free_all_blocks()

    # 下采样场景面
    num_faces = 1000000
    downsampled_faces = downsample_faces(scene_faces, num_faces)
    downsampled_vertices, downsampled_faces = update_vertices_faces(
        scene_vertices, downsampled_faces
    )
    logging.info(f"downsample completed")
    # 仅释放原始场景数据（保留downsampled_vertices和downsampled_faces）
    del scene_vertices, scene_faces
    cp.get_default_memory_pool().free_all_blocks()

    # 读取放置点数据
    place_points_list, path_points_list, camera_points_list = read_place_txt(txt_file)
    place_points = cp.array(place_points_list, dtype=cp.float32)
    path_points = cp.array(path_points_list, dtype=cp.float32)
    camera_points = cp.array(camera_points_list, dtype=cp.float32)
    del place_points_list, path_points_list, camera_points_list
    cp.get_default_memory_pool().free_all_blocks()

    steps = 2880

    # 输出目录设置
    signal_all_dir = os.path.join(out_dir, "signal_all")

    os.makedirs(signal_all_dir, exist_ok=True)

    # 处理单个视角的函数
    def process_signal(index):
        cam_point = camera_points[index]
        if index == 0:
            start_path = path_points[0]
            end_path = path_points[1]
        elif index == 1:
            start_path = path_points[1]
            end_path = path_points[0]
        elif index == 2:
            start_path = path_points[3]
            end_path = path_points[2]
        else:
            start_path = path_points[2]
            end_path = path_points[3]

        logging.info(f"Starting 视角 {index + 1} 处理")

        # 计算信号（确保所有变量都有效）
        all_signals = object_move_signal_simulate(
            item_vertices, item_faces,
            downsampled_vertices, downsampled_faces,
            mask_vertices, mask_faces,
            cam_point, place_points,
            start_path, end_path, steps
        )

        # 保存结果并清理
        save_cupy_array_to_npy(
            all_signals,
            os.path.join(signal_all_dir, f"signal_all_{index + 1}.npy")
        )

        del all_signals, cam_point, start_path, end_path
        cp.get_default_memory_pool().free_all_blocks()
        logging.info(f"视角 {index + 1} 处理完成")

    # 依次处理四个视角
    process_signal(0)
    process_signal(1)
    process_signal(2)
    process_signal(3)

    # 最终清理所有全局变量
    del downsampled_vertices, downsampled_faces, item_vertices, item_faces, mask_vertices, mask_faces, place_points, path_points, camera_points
    cp.get_default_memory_pool().free_all_blocks()
    logging.info(f"All perspectives processed, memory fully released")
