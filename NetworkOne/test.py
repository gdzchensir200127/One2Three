import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Tuple, Optional
from torch.cuda.amp import autocast
from net12 import Fnet  
import psutil


# -------------------------- 1. 配置日志 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# -------------------------- 2. 数据加载类--------------------------
class SignalDepthTestDataset(Dataset):
    """测试专用数据集"""
    def __init__(
        self,
        base_dir: str,
        sample_ids: List[str],
        signal_temporal_dim: int = 2880,
        depth_temporal_dim: int = 720,
        signal_dim: int = 96,
    ):
        self.base_dir = Path(base_dir)
        self.sample_ids = sample_ids  # 所有测试样本ID
        self.depth_temporal_dim = depth_temporal_dim
        self.signal_temporal_dim = signal_temporal_dim
        self.signal_dim = signal_dim

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_signal_npy(self, file_path: Path) -> np.ndarray:
        if not file_path.exists():
            raise FileNotFoundError(f"信号文件不存在:{file_path}")
        signal = np.load(file_path)
        logger.debug(f"信号文件 {file_path.name} 范围: {signal.min():.4f} ~ {signal.max():.4f}")
        
        # 数据校验
        if np.isnan(signal).any() or np.isinf(signal).any():
            raise ValueError(f"信号文件{file_path}包含nan/inf!")
        expected_shape = (self.signal_temporal_dim, self.signal_dim)
        assert signal.shape == expected_shape, \
            f"信号切片形状错误: 期望{expected_shape}, 实际{signal.shape}"
        assert signal.dtype == np.float32, \
            f"信号数据类型错误: 期望float32, 实际{signal.dtype}"
        return signal

    def _load_depth_h5(self, file_path: Path) -> np.ndarray:
        if not file_path.exists():
            raise FileNotFoundError(f"深度文件不存在:{file_path}")
        with h5py.File(file_path, "r") as f:
            depth = f["depth_data"][:]  
        logger.debug(f"深度文件 {file_path.name} 范围: {depth.min():.4f} ~ {depth.max():.4f}")
        
        # 数据校验
        if np.isnan(depth).any() or np.isinf(depth).any():
            raise ValueError(f"深度文件{file_path}包含nan/inf!")
        expected_shape = (self.depth_temporal_dim, 576, 640)
        assert depth.shape == expected_shape, \
            f"深度图切片形状错误: 期望{expected_shape}, 实际{depth.shape}"
        assert depth.dtype == np.float32 and 0.0 <= depth.min() <= depth.max() <= 1.0, \
            "深度图数据格式或范围错误"
        return depth

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        测试时仅需输入数据(信号+深度)和样本ID
        返回: (signals, depths, sample_id)
        """
        sample_id = self.sample_ids[idx]

        # 加载输入信号(实部+虚部,与训练输入格式一致)
        signals = np.zeros((self.signal_temporal_dim, 2, self.signal_dim), dtype=np.float32)
        real_input = self._load_signal_npy(self.base_dir / "signal_norm_all_real" / f"{sample_id}.npy")
        imag_input = self._load_signal_npy(self.base_dir / "signal_norm_all_imag" / f"{sample_id}.npy")
        signals[:, 0, :] = real_input  # 实部通道
        signals[:, 1, :] = imag_input  # 虚部通道

        # 加载深度数据
        depths = self._load_depth_h5(self.base_dir / "depth_all_extra" / f"{sample_id}.h5")

        # 转换为Tensor并返回(含样本ID)
        return (
            torch.from_numpy(signals),
            torch.from_numpy(depths),
            sample_id  # 返回样本ID用于结果命名
        )


# -------------------------- 3. 数据预处理工具 --------------------------
def collect_all_test_samples(base_dir: str) -> List[str]:
    """收集所有有效测试样本"""
    base_dir = Path(base_dir)
    signal_real_dir = base_dir / "signal_norm_all_real"
    if not signal_real_dir.exists():
        raise NotADirectoryError(f"信号目录不存在:{signal_real_dir}")
    
    # 从实部信号目录获取所有候选样本ID
    sample_id_set = {
        os.path.splitext(file_name)[0] 
        for file_name in os.listdir(signal_real_dir) 
        if file_name.endswith(".npy")
    }
    
    # 校验样本完整性
    valid_sample_ids = []
    required_files = [
        ("signal_norm_all_real", ".npy"),
        ("signal_norm_all_imag", ".npy"),
        ("depth_all_extra", ".h5")
    ]
    
    for sample_id in sample_id_set:
        is_valid = True
        for dir_name, ext in required_files:
            file_path = base_dir / dir_name / f"{sample_id}{ext}"
            if not file_path.exists():
                logger.warning(f"样本{sample_id}缺失{dir_name}/{sample_id}{ext},跳过")
                is_valid = False
                break
        if is_valid:
            valid_sample_ids.append(sample_id)
    
    if len(valid_sample_ids) == 0:
        raise ValueError("未找到任何有效测试样本")
    logger.info(f"共收集到 {len(valid_sample_ids)} 个有效测试样本")
    return valid_sample_ids


# -------------------------- 4. 模型加载工具 --------------------------
def load_test_model(
    checkpoint_path: str,
    gpu_ids: List[int],
    device: torch.device
) -> nn.Module:
    """加载训练好的模型权重"""
    # 初始化模型
    model = Fnet().to(device)
    
    # 加载Checkpoint
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理多GPU训练的权重前缀
    state_dict = checkpoint["model_state_dict"]
    if isinstance(model, nn.DataParallel) or ("module." in next(iter(state_dict.keys()))):
        # 若当前单GPU/CPU测试,移除权重中的"module."前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # 移除"module."
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # 加载权重
    model.load_state_dict(state_dict)
    logger.info(f"成功从 {checkpoint_path} 加载模型权重(训练epoch: {checkpoint['epoch']})")
    
    # 多GPU测试支持
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"启用多GPU测试,设备列表: {gpu_ids}")
    
    # 切换为评估模式
    model.eval()
    return model


# -------------------------- 5. 结果保存工具 --------------------------
def init_result_dirs(result_root: str = " ") -> Tuple[Path, Path]:
    """初始化结果保存目录(real/实部, imag/虚部)"""
    real_dir = Path(result_root) / "real"
    imag_dir = Path(result_root) / "imag"
    
    # 创建目录
    for dir_path in [real_dir, imag_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"结果保存目录初始化完成: {dir_path}")
    
    return real_dir, imag_dir


def save_prediction(
    pred_real: torch.Tensor,
    pred_imag: torch.Tensor,
    sample_id: str,
    real_dir: Path,
    imag_dir: Path
) -> None:
    """保存单样本的预测结果为npy文件"""
    # Tensor -> Numpy(保持float32类型)
    pred_real_np = pred_real.cpu().numpy().astype(np.float32)
    pred_imag_np = pred_imag.cpu().numpy().astype(np.float32)
    
    # 保存路径
    real_save_path = real_dir / f"{sample_id}.npy"
    imag_save_path = imag_dir / f"{sample_id}.npy"
    
    # 保存文件
    np.save(real_save_path, pred_real_np)
    np.save(imag_save_path, pred_imag_np)
    
    # 日志记录
    logger.debug(f"样本 {sample_id} 保存完成:")
    logger.debug(f"  实部: {real_save_path} (形状: {pred_real_np.shape}, 范围: {pred_real_np.min():.4f}~{pred_real_np.max():.4f})")
    logger.debug(f"  虚部: {imag_save_path} (形状: {pred_imag_np.shape}, 范围: {pred_imag_np.min():.4f}~{pred_imag_np.max():.4f})")


# -------------------------- 6. 测试核心函数 --------------------------
def test(
    base_dir: str,
    checkpoint_path: str,
    result_root: str = " ",
    batch_size: int = 2,
    gpu_ids: List[int] = [4, 5],
    num_workers: int = 4
) -> None:
    """
    完整测试流程:加载数据→加载模型→推理→保存结果
    Args:
        base_dir: 测试样本根目录
        checkpoint_path: 训练好的模型权重路径
        result_root: 结果保存根目录
        batch_size: 测试批次大小
        gpu_ids: 可用GPU编号列表
        num_workers: 数据加载线程数
    """
    # -------------------------- 步骤1:初始化设备 --------------------------
    if len(gpu_ids) == 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("未使用GPU,将使用CPU进行测试(推理速度较慢)")
    else:
        # 校验GPU有效性
        available_gpus = torch.cuda.device_count()
        valid_gpus = [g for g in gpu_ids if 0 <= g < available_gpus]
        if not valid_gpus:
            device = torch.device("cpu")
            logger.warning(f"指定GPU {gpu_ids} 不可用,切换为CPU测试")
        else:
            device = torch.device(f"cuda:{valid_gpus[0]}")
            torch.cuda.set_device(device)
            gpu_ids = valid_gpus
    logger.info(f"测试设备初始化完成: {device} (可用GPU: {gpu_ids})")

    # -------------------------- 步骤2:初始化结果目录 --------------------------
    real_dir, imag_dir = init_result_dirs(result_root)

    # -------------------------- 步骤3:加载测试数据 --------------------------
    logger.info("开始加载测试数据...")
    sample_ids = collect_all_test_samples(base_dir)
    
    # 创建测试数据集
    test_dataset = SignalDepthTestDataset(
        base_dir=base_dir,
        sample_ids=sample_ids
    )
    
    # 创建测试数据加载器(禁用shuffle和drop_last,确保所有样本被处理)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2
    )
    logger.info(f"测试数据加载完成:共 {len(test_dataset)} 个样本,{len(test_loader)} 个批次")

    # -------------------------- 步骤4:加载模型 --------------------------
    logger.info("开始加载模型...")
    model = load_test_model(checkpoint_path, gpu_ids, device)
    logger.info("模型加载完成,进入推理模式")

    # -------------------------- 步骤5:资源监控(可选) --------------------------
    ram = psutil.virtual_memory()
    logger.info(f"测试前系统资源:RAM已用 {ram.used/1024**3:.2f}GB / 总 {ram.total/1024**3:.2f}GB")
    for gpu in gpu_ids:
        with torch.cuda.device(gpu):
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU {gpu} 初始显存占用:{mem:.2f}GB")

    # -------------------------- 步骤6:推理与结果保存 --------------------------
    logger.info("="*50)
    logger.info("开始测试推理...")
    pbar = tqdm(test_loader, desc="Test Inference")
    
    # 禁用梯度计算
    with torch.no_grad():
        for batch_idx, (signals, depths, batch_sample_ids) in enumerate(pbar):
            # 数据转移到设备
            signals = signals.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            
            # 混合精度推理
            with torch.amp.autocast(device_type='cuda'):
                pred_real, pred_imag = model(signals, depths)  # 模型输出:实部+虚部
            
            # 逐个样本保存结果(batch_sample_ids与预测结果一一对应)
            for i in range(len(batch_sample_ids)):
                save_prediction(
                    pred_real=pred_real[i],  # 第i个样本的实部预测
                    pred_imag=pred_imag[i],  # 第i个样本的虚部预测
                    sample_id=batch_sample_ids[i],  # 第i个样本的ID
                    real_dir=real_dir,
                    imag_dir=imag_dir
                )
            
            # 更新进度条信息
            pbar.set_postfix({"已处理样本": (batch_idx + 1) * batch_size})
            
            # 清理显存(避免累积)
            torch.cuda.empty_cache()

    # -------------------------- 测试完成 --------------------------
    logger.info("="*50)
    logger.info("所有测试样本推理完成！")
    logger.info(f"实部结果保存路径:{real_dir}")
    logger.info(f"虚部结果保存路径:{imag_dir}")
    logger.info("="*50)


# -------------------------- 7. 主函数 --------------------------
if __name__ == "__main__":
    # 环境变量配置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 测试参数配置(根据实际情况调整)
    TEST_CONFIG = {
        "base_dir": " ",  # 测试样本根目录
        "checkpoint_path": " ",  # 训练好的模型权重
        "result_root": " ",  # 结果保存根目录
        "batch_size": 2,  # 批次大小
        "gpu_ids": [3],  # 测试用GPU
        "num_workers": 16  # 数据加载线程数
    }
    
    # 启动测试
    test(**TEST_CONFIG)