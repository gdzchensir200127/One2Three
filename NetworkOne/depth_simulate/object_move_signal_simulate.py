import numpy as np
import os
import gc
from depth_simulate import depth_simulate
from util.read_file import read_place_txt, parse_scene_ply, parse_item_ply
from util.util import place_mesh, downsample_faces, update_vertices_faces, \
    filter_faces_by_distance
from util.util import save_numpy_depths_to_pngs

from concurrent.futures import as_completed, ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')


def process_step(i: int, vertices: np.ndarray, item_faces: np.ndarray,
                 rest_vertices: np.ndarray, rest_faces: np.ndarray,
                 camera_points: np.ndarray, init_place_points: np.ndarray,
                 place_points_list: np.ndarray, item_botton_center: np.ndarray,
                 single_depth_dir: str, all_depth_dir: str) -> None:
    """处理单个step的并行任务"""
    try:
        # 计算当前step的物体偏移（向量化操作）
        place_points = place_points_list[i]
        translation = place_points - item_botton_center
        item_vertices_offset = vertices.copy()
        item_vertices_offset[:, :3] += translation  # 直接对数组进行偏移操作

        # 合并物体与场景
        merged_vertices, merged_faces = place_mesh(
            place_points, rest_vertices, rest_faces,
            vertices, item_faces
        )

        # 定义当前step的两个子任务
        def task1() -> None:
            depth = depth_simulate(
                item_vertices_offset, item_faces,  # 直接传入numpy数组
                camera_points, init_place_points
            )
            save_numpy_depths_to_pngs(depth, single_depth_dir, i)
            del depth
            gc.collect()

        def task2() -> None:
            depth = depth_simulate(
                merged_vertices, merged_faces,
                camera_points, init_place_points
            )
            save_numpy_depths_to_pngs(depth, all_depth_dir, i)
            del depth
            gc.collect()

        # 并行执行当前step的两个任务
        with ThreadPoolExecutor(max_workers=2) as step_executor:
            futures = [
                step_executor.submit(task1),
                step_executor.submit(task2),
            ]
            # 等待所有子任务完成
            for future in as_completed(futures):
                if future.exception():
                    logging.error(f"Step {i} task error: {future.exception()}")

        # 清理当前step的临时变量
        del merged_vertices, merged_faces, item_vertices_offset
        gc.collect()
        logging.info(f"Completed step {i}/{len(place_points_list) - 1}")

    except Exception as e:
        logging.error(f"Error in step {i}: {str(e)}")
        import traceback
        traceback.print_exc()


def object_move_signal_simulate(
        item_vertices: np.ndarray, item_faces: np.ndarray,
        scene_vertices: np.ndarray, scene_faces: np.ndarray,
        camera_points: np.ndarray,
        init_place_points: np.ndarray,
        start_path_points: np.ndarray,
        end_path_points: np.ndarray,
        steps: int,
        single_depth_dir: str,
        all_depth_dir: str
) -> None:
    """
    模拟物体在多个移动步长下的雷达信号变化（并行化版本）
    """
    # 预处理顶点数据（保持原有逻辑，但使用numpy操作）
    x_coords = item_vertices[:, 0]
    y_coords = item_vertices[:, 1]
    z_coords = item_vertices[:, 2]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_z = np.min(z_coords)
    item_botton_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, min_z])

    # 计算所有偏移位置（向量化生成）
    place_points_list = np.linspace(start_path_points, end_path_points, steps)

    # 预处理场景数据
    distance_threshold = 6.0
    rest_vertices, rest_faces = filter_faces_by_distance(
        scene_vertices, scene_faces, camera_points, distance_threshold
    )
    rest_vertices, rest_faces = update_vertices_faces(rest_vertices, rest_faces)

    logging.info(f"Starting parallel processing for {steps} steps")

    # 并行处理所有steps
    max_parallel_steps = min(1, steps)  # 避免线程过多导致资源耗尽
    with ThreadPoolExecutor(max_workers=max_parallel_steps) as executor:
        # 为每个step创建一个任务
        futures = [
            executor.submit(
                process_step,
                i, item_vertices, item_faces, rest_vertices, rest_faces,
                camera_points, init_place_points, place_points_list,
                item_botton_center, single_depth_dir, all_depth_dir,
            ) for i in range(steps)
        ]

        # 等待所有step完成
        for future in as_completed(futures):
            if future.exception():
                logging.error(f"Step task failed: {future.exception()}")

    logging.info(f"All {steps} steps completed")


def both_object_move_signal_simulate(txt_file, scene_file, item_file, out_dir):
    # 保持原有逻辑不变，确保输出正确
    scene_vertices, scene_faces = parse_scene_ply(scene_file)
    item_vertices, item_faces = parse_item_ply(item_file)
    scene_vertices = np.array(scene_vertices, dtype=np.float32)
    scene_faces = np.array(scene_faces, dtype=np.int32)
    item_vertices = np.array(item_vertices, dtype=np.float32)
    item_faces = np.array(item_faces, dtype=np.int32)

    # 下采样场景面
    num_faces = 1000000
    downsampled_faces = downsample_faces(scene_faces, num_faces)
    downsampled_vertices, downsampled_faces = update_vertices_faces(
        scene_vertices, downsampled_faces
    )

    place_points, path_points, camera_points = read_place_txt(txt_file)
    place_points = np.array(place_points, dtype=np.float32)
    path_points = np.array(path_points, dtype=np.float32)
    camera_points = np.array(camera_points, dtype=np.float32)
    steps = 2880

    # 输出目录设置
    signal_single_dir = os.path.join(out_dir, "signal_single")
    signal_all_dir = os.path.join(out_dir, "signal_all")
    single_depth_dir = os.path.join(out_dir, "single_depth")
    all_depth_dir = os.path.join(out_dir, "all_depth")

    # 创建目录
    for dir_path in [signal_single_dir, signal_all_dir, single_depth_dir, all_depth_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 四个视角的目录
    single_depth_dirs = [
        os.path.join(single_depth_dir, f"single_depth_{i + 1}")
        for i in range(4)
    ]
    all_depth_dirs = [
        os.path.join(all_depth_dir, f"all_depth_{i + 1}")
        for i in range(4)
    ]

    for dir_path in single_depth_dirs + all_depth_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 并行处理四个视角
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

        logging.info(f"Starting视角 {index + 1} 处理")
        object_move_signal_simulate(
            item_vertices, item_faces,
            downsampled_vertices, downsampled_faces,
            cam_point, place_points,
            start_path, end_path, steps,
            single_depth_dirs[index], all_depth_dirs[index]
        )

        gc.collect()
        logging.info(f"视角 {index + 1} 处理完成")

    # 并行处理四个视角
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_signal, range(4))