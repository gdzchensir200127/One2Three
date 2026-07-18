import cupy as cp
import time
from multiprocessing import Process, Queue, set_start_method, Lock, Manager
from signal_simulate.simulate.simulate_moving_object_cupy1 import simulate
import os
from tqdm import tqdm

# 数据集路径和输出路径
dataset_path = "/home/zhang_muxin/shapenetS2M_HPR"
output_root = "/home/zhang_muxin/shapenetS2M_simulate"
obj_names = ["0.obj", "1.obj", "2.obj", "3.obj"]

# 处理单个文件的函数
def process_file(dir_path: str):
    ret = []
    for obj_name in obj_names:
        input_path = os.path.join(dir_path, obj_name)
        output_path = input_path.replace(dataset_path, output_root).replace('.obj', '.mat')
        output_path_dir = os.path.dirname(output_path)
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)
        ret.append((input_path, output_path))
    return ret

# 每个 GPU 的最大并行任务数
MAX_TASKS_PER_GPU = {
    0: 10,  # GPU 0 最多同时运行 10 个任务
    1: 10,  # GPU 1 最多同时运行 10 个任务
    2: 10,  # GPU 2 最多同时运行 10 个任务
    3: 6,  # GPU 3 最多同时运行 6 个任务
}

# 任务函数
def task(task_id, gpu_id, result_queue, path_pair):
    # 选择指定的 GPU
    device = cp.cuda.Device(gpu_id)
    device.use()
    
    # 创建一个新的 CUDA 流
    stream = cp.cuda.Stream()
    with stream:
        # 处理文件
        simulate(path_pair[0], path_pair[1])
        cp.cuda.stream.get_current_stream().synchronize()
        
        # 将结果放入队列
        result_queue.put((gpu_id, f"Task {task_id} done on GPU {gpu_id} {path_pair}"))

def process_category(category: str):
    root_dir = os.path.join(dataset_path, category)
    dir_list = []
    for dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir)):
            obj_path = os.path.join(root_dir, dir, 'models')
            dir_list.append(obj_path)
    dir_list.sort()
    file_list = []
    for dir in dir_list:
        for file in process_file(dir):
            file_list.append(file)

    # 记录开始时间
    start_time = time.time()
    
    # 创建结果队列和锁
    result_queue = Queue()
    lock = Lock()
    
    # 使用 Manager 共享任务计数
    manager = Manager()
    gpu_task_count = manager.dict({gpu_id: 0 for gpu_id in MAX_TASKS_PER_GPU})
    
    # 创建并启动进程
    processes = []
    task_id = 0
    while task_id < len(file_list):  # 遍历所有文件
        with lock:
            # 找到一个可用的 GPU
            available_gpu = min(MAX_TASKS_PER_GPU.keys(), key=lambda x: gpu_task_count[x])
            if gpu_task_count[available_gpu] < MAX_TASKS_PER_GPU[available_gpu]:
                # 启动任务
                p = Process(target=task, args=(task_id, available_gpu, result_queue, file_list[task_id]))
                p.start()
                processes.append((p, available_gpu))
                gpu_task_count[available_gpu] += 1
                task_id += 1
                print(f'task id: {task_id}, total: {len(file_list)}, assigned to GPU {available_gpu}\n')
        
        # 检查已完成的任务
        for p, gpu_id in processes[:]:
            p.join(0.1)  # 非阻塞等待
            if not p.is_alive():
                processes.remove((p, gpu_id))
                with lock:
                    gpu_task_count[gpu_id] -= 1
                print(f"Task on GPU {gpu_id} completed. Remaining tasks: {gpu_task_count[gpu_id]}")
    
    # 等待所有进程完成
    for p, _ in processes:
        p.join()
    
    # 记录结束时间
    end_time = time.time()
    
    # 输出结果
    while not result_queue.empty():
        gpu_id, result = result_queue.get()
        print(result)
    
    # 输出总耗时
    print(f"Total time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    # 使用 spawn 启动方法
    set_start_method('spawn')
    # process_category("扳手")
    # process_category("笔记本电脑")
    # process_category("刀具")
    # process_category("键盘")
    # process_category("手机")
    # process_category("水杯")
    # process_category("显示器")
    process_category("03211117")
    process_category("03636649")
    process_category("03691459")
    process_category("04090263")

    process_category("04256520")

    process_category("04401088")

    process_category("04379243")

    process_category("02691156")

    process_category("04530566")
    process_category("02828884")
    process_category("02933112")

    # process_category("02958343")
    # process_category("03001627")