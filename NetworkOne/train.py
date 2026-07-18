import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR  
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import h5py
import os
from pathlib import Path
import random
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional, Union
from torch.cuda.amp import GradScaler, autocast
from net12 import Fnet  
import psutil


# -------------------------- 配置日志 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# -------------------------- Early Stopping 类定义 --------------------------
class EarlyStopping:
    """早停"""
    def __init__(
        self,
        patience: int = 10,          
        min_delta: float = 1e-6,    
        verbose: bool = True        
    ):
        self.patience = patience          
        self.min_delta = min_delta        
        self.verbose = verbose            
        self.counter = 0                  
        self.best_score = None            
        self.early_stop = False           

    def step(self, current_val_loss: float) -> None:
        if self.best_score is None:
            self.best_score = current_val_loss
            if self.verbose:
                logger.info(f"初始化最佳验证损失: {self.best_score:.6f}")
        
        elif current_val_loss < (self.best_score - self.min_delta):
            self.best_score = current_val_loss
            self.counter = 0
            if self.verbose:
                logger.info(f"验证损失改善至: {self.best_score:.6f}，重置早停计数器")
        
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"早停计数器: {self.counter}/{self.patience} (当前验证损失: {current_val_loss:.6f}, 最佳验证损失: {self.best_score:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.warning(f"早停条件触发！连续{self.patience}个epoch验证损失无有效改善,停止训练")


# -------------------------- 数据加载类 --------------------------
class SignalDepthDataset(Dataset):
    def __init__(
        self,
        base_dir:str,
        sample_ids:List[str],
        signal_temporal_dim:int = 2880,
        depth_temporal_dim:int = 720,
        signal_dim:int = 96,
    ):
        self.base_dir = Path(base_dir)
        self.sample_ids = sample_ids
        self.depth_temporal_dim = depth_temporal_dim
        self.signal_temporal_dim = signal_temporal_dim
        self.signal_dim = signal_dim
        
        self.all_indices = []
        for sample_id in self.sample_ids:
            self.all_indices.append(sample_id)

    def __len__(self) -> int:
        return len(self.all_indices)

    def _load_signal_npy(self, file_path: Path) -> np.ndarray:
        if not file_path.exists():
            raise FileNotFoundError(f"信号文件不存在:{file_path}")
        f = np.load(file_path)
        signal = f[:]
        logger.debug(f"信号文件 {file_path.name} 范围: {signal.min():.4f} ~ {signal.max():.4f}")
        if np.isnan(signal).any() or np.isinf(signal).any():
            raise ValueError(f"信号文件{file_path}包含nan/inf!")
        expected_shape = (2880, 96)
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
        if np.isnan(depth).any() or np.isinf(depth).any():
            raise ValueError(f"深度文件{file_path}包含nan/inf!")
        expected_shape = (720, 576, 640)
        assert depth.shape == expected_shape, \
            f"深度图切片形状错误: 期望{expected_shape}, 实际{depth.shape}"
        assert depth.dtype == np.float32 and 0.0 <= depth.min() <= depth.max() <= 1.0, \
            "深度图数据格式或范围错误"
        return depth

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_id = self.all_indices[idx]

        signals = np.zeros((self.signal_temporal_dim, 2, self.signal_dim), dtype=np.float32)
        depths = np.zeros((self.depth_temporal_dim, 576, 640), dtype=np.float32)
        gt_real = np.zeros((self.signal_temporal_dim, self.signal_dim), dtype=np.float32)
        gt_imag = np.zeros((self.signal_temporal_dim, self.signal_dim), dtype=np.float32)

        real_input = self._load_signal_npy(
                self.base_dir / "signal_norm_all_real" / f"{sample_id}.npy",
            )
        signals[ :, 0, :] = real_input

        imag_input = self._load_signal_npy(
                self.base_dir / "signal_norm_all_imag" / f"{sample_id}.npy",
            )
        signals[ :, 1, :] = imag_input

        depth_data = self._load_depth_h5(
                self.base_dir / "depth_all_extra" / f"{sample_id}.h5",
            )
        depths[ :, :, :] = depth_data

        gt_real = self._load_signal_npy(
                self.base_dir / "signal_norm_single_real" / f"{sample_id}.npy",
            )
        gt_imag = self._load_signal_npy(
                self.base_dir / "signal_norm_single_imag" / f"{sample_id}.npy",
            )

        return (
            torch.from_numpy(signals),
            torch.from_numpy(depths),
            torch.from_numpy(gt_real),
            torch.from_numpy(gt_imag)
        )


# -------------------------- 数据预处理工具 --------------------------
def collect_sample_ids(base_dir:str) -> List[str]:
    base_dir = Path(base_dir)
    signal_real_dir = base_dir / "signal_norm_all_real"
    if not signal_real_dir.exists():
        raise NotADirectoryError(f"信号目录不存在:{signal_real_dir}")
    
    sample_id_set = set()
    for file_name in os.listdir(signal_real_dir):
        if file_name.endswith(".npy"):
            prefix = os.path.splitext(file_name)[0]
            sample_id_set.add(prefix)
    
    valid_sample_ids = []
    required_dirs = [
        "signal_norm_all_real", "signal_norm_all_imag", 
        "signal_norm_single_real", "signal_norm_single_imag", 
        "depth_all_extra"
    ]
    
    for sample_id in sample_id_set:
        is_valid = True
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            ext = ".npy" if "signal" in dir_name else ".h5"
            file_path = dir_path / f"{sample_id}{ext}"
            if not file_path.exists():
                logger.warning(f"样本{sample_id}的{dir_name}/{sample_id}{ext}缺失,跳过该样本")
                is_valid = False
                break
            if not is_valid:
                break
        if is_valid:
            valid_sample_ids.append(sample_id)
    
    if len(valid_sample_ids) == 0:
        raise ValueError("没有找到有效的样本")
    logger.info(f"收集到{len(valid_sample_ids)}个有效样本")
    return valid_sample_ids

def split_train_val(sample_ids:List[str], val_ratio:float = 0.2, seed:int = 42) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    random.shuffle(sample_ids)
    split_idx = int(len(sample_ids) * (1 - val_ratio))
    train_ids = sample_ids[:split_idx]
    val_ids = sample_ids[split_idx:]
    logger.info(f"训练集样本数:{len(train_ids)}, 验证集样本数:{len(val_ids)}")
    return train_ids, val_ids


# -------------------------- TensorBoard权重/梯度记录工具函数 --------------------------
def log_model_params_to_tb(writer: SummaryWriter, model: nn.Module, step: int, log_type: str = "weights"):
    current_model = model.module if isinstance(model, nn.DataParallel) else model
    
    for name, param in current_model.named_parameters():
        if not param.requires_grad:
            continue
        
        tag = f"Model/{log_type}/{name.replace('.', '/')}"
        
        if log_type == "weights":
            writer.add_histogram(tag, param.data, step)
        
        elif log_type == "grads":
            if param.grad is not None:
                grad_l2_norm = torch.norm(param.grad, p=2)
                writer.add_scalar(tag + "_L2_Norm", grad_l2_norm, step)
            else:
                writer.add_scalar(tag + "_L2_Norm", 0.0, step)


# -------------------------- 训练核心函数 --------------------------
def train_one_epoch(
    model:nn.Module,
    dataloader:DataLoader,
    optimizer:optim.Optimizer,
    scheduler:optim.lr_scheduler.OneCycleLR,  
    device:torch.device,
    epoch:int,
    writer:SummaryWriter,
    global_step:int,
    scaler: GradScaler,
    grad_clip: float = 1.0  
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Train)")
    
    for batch_idx, (signals, depths, gt_real, gt_imag) in enumerate(pbar):
        signals = signals.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        gt_real = gt_real.to(device, non_blocking=True)
        gt_imag = gt_imag.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 前向传播(混合精度)
        with torch.amp.autocast(device_type='cuda'):
            pred_real, pred_imag = model(signals, depths)
            cos_sim_real = F.cosine_similarity(pred_real, gt_real, dim=-1)
            cos_sim_imag = F.cosine_similarity(pred_imag, gt_imag, dim=-1)
            cos_sim_real=1-cos_sim_real
            cos_sim_imag=1-cos_sim_imag
            cos_sim_real=cos_sim_real.mean()
            cos_sim_imag=cos_sim_imag.mean()
            total_batch_loss=0.25*(cos_sim_real+cos_sim_imag)

        # 反向传播(混合精度)
        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(optimizer)
        
        # # 记录梯度
        # log_model_params_to_tb(writer, model, global_step, log_type="grads")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        grad_norms = [torch.norm(p.grad, p=2) for p in model.parameters() if p.grad is not None]
        if grad_norms:
            logger.debug(f"Batch {batch_idx} 梯度L2范数范围: {min(grad_norms):.4f} ~ {max(grad_norms):.4f}")

        # 更新优化器和scaler
        scaler.step(optimizer)
        scaler.update()
        
        # 每个batch后调用OneCycleLR的step()
        scheduler.step()

        # 损失统计和TensorBoard记录
        total_loss += total_batch_loss.item() * signals.size(0)
        pbar.set_postfix({"Batch Loss": f"{total_batch_loss.item():.4f}"})
        writer.add_scalar("Train/Batch Loss", total_batch_loss.item(), global_step)
        global_step += 1
    
    avg_loss = total_loss / len(dataloader.dataset)
    logger.info(f"Epoch {epoch} - Train Loss:{avg_loss:.4f}")
    return avg_loss, global_step

def validate_one_epoch(
    model:nn.Module,
    dataloader:DataLoader,
    device:torch.device,
    epoch:int,
    gpu_ids:List[int]
) -> float:
    ram = psutil.virtual_memory()
    logger.info(f"验证前RAM使用:已用{ram.used/1024**3:.2f}GB / 总{ram.total/1024**3:.2f}GB (使用率{ram.percent}%)")
    for gpu_id in gpu_ids:
        with torch.cuda.device(gpu_id):
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU {gpu_id} 显存:已分配{mem_allocated:.2f}GB / 已预留{mem_reserved:.2f}GB")

    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Val)")
    
    with torch.no_grad():
        for batch_idx, (signals, depths, gt_real, gt_imag) in enumerate(pbar):
            signals = signals.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            gt_real = gt_real.to(device, non_blocking=True)
            gt_imag = gt_imag.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda'):
                pred_real, pred_imag = model(signals, depths)
                cos_sim_real = F.cosine_similarity(pred_real, gt_real, dim=-1)
                cos_sim_imag = F.cosine_similarity(pred_imag, gt_imag, dim=-1)
                cos_sim_real=1-cos_sim_real
                cos_sim_imag=1-cos_sim_imag
                cos_sim_real=cos_sim_real.mean()
                cos_sim_imag=cos_sim_imag.mean()
                total_batch_loss=0.25*(cos_sim_real+cos_sim_imag)
                
            total_loss += total_batch_loss.item() * signals.size(0)
            pbar.set_postfix({"Batch Loss":f"{total_batch_loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader.dataset)
    logger.info(f"Epoch {epoch} - Val Loss:{avg_loss:.4f}")
    return avg_loss

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.OneCycleLR,
    scaler: GradScaler,
    early_stopper: EarlyStopping,  
    epoch: int,
    global_step: int,
    best_val_loss: float,
    save_dir: Path,
    is_best: bool = False
) -> None:
    current_model = model.module if isinstance(model, nn.DataParallel) else model
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': current_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stopper_state': {
            'counter': early_stopper.counter,
            'best_score': early_stopper.best_score,
            'early_stop': early_stopper.early_stop
        }
    }
    
    # 保存常规检查点
    checkpoint_path = save_dir / f"checkpoint_epoch{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"已保存检查点到: {checkpoint_path}")
    
    # 保存最佳模型
    if is_best:
        best_checkpoint_path = save_dir / "best_checkpoint.pth"
        torch.save(checkpoint, best_checkpoint_path)
        logger.info(f"已保存最佳检查点到: {best_checkpoint_path}")
    
    # 清理旧检查点(仅保留最近1个)
    epoch_checkpoints = [f for f in save_dir.glob("checkpoint_epoch*.pth") if f.is_file()]
    # 按epoch号排序（确保最新的在最后）
    epoch_checkpoints.sort(key=lambda x: int(x.stem.split("epoch")[-1]))
    
    # 若检查点数量超过1，删除所有旧检查点，仅保留最新的1个
    if len(epoch_checkpoints) > 1:
        # 保留最后1个，删除前面所有
        for old_ckpt in epoch_checkpoints[:-1]:
            old_ckpt.unlink()
            logger.info(f"已删除旧检查点(仅保留最近1个): {old_ckpt.name}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.OneCycleLR,  # 更新scheduler类型
    scaler: GradScaler,
    early_stopper: EarlyStopping,
    device: torch.device
) -> Tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载状态
    current_model = model.module if isinstance(model, nn.DataParallel) else model
    current_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 恢复OneCycleLR状态
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # 恢复早停状态
    if 'early_stopper_state' in checkpoint:
        early_stopper.counter = checkpoint['early_stopper_state']['counter']
        early_stopper.best_score = checkpoint['early_stopper_state']['best_score']
        early_stopper.early_stop = checkpoint['early_stopper_state']['early_stop']
        logger.info(f"已恢复早停状态: 计数器={early_stopper.counter}, 最佳分数={early_stopper.best_score:.6f}")
    
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    best_val_loss = checkpoint['best_val_loss']
    
    logger.info(f"已从检查点恢复训练状态:  epoch={epoch}, global_step={global_step}, best_val_loss={best_val_loss:.4f}")
    return epoch, global_step, best_val_loss

def train(
    base_dir:str,
    epochs:int = 50,
    batch_size:int = 8,
    lr:float = 1e-4,
    val_ratio:float = 0.2,
    save_dir:str = "saved_models",
    seed:int = 42,
    gpu_ids:List[int] = [0],
    grad_clip: float = 1.0,  
    resume_checkpoint: Optional[str] = None,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 1e-6
):
    # 初始化随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设备选择
    if len(gpu_ids) == 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("未指定GPU或无可用GPU,将使用CPU训练")
    else:
        available_gpu_count = torch.cuda.device_count()
        valid_gpu_ids = [gpu for gpu in gpu_ids if 0 <= gpu < available_gpu_count]
        invalid_gpu_ids = [gpu for gpu in gpu_ids if gpu not in valid_gpu_ids]
        if invalid_gpu_ids:
            logger.warning(f"指定的GPU编号{invalid_gpu_ids}不存在(可用GPU数量:{available_gpu_count}),已自动过滤")
        if len(valid_gpu_ids) == 0:
            device = torch.device("cpu")
            logger.warning("无有效GPU,将使用CPU训练")
        else:
            device = torch.device(f"cuda:{valid_gpu_ids[0]}")
            torch.cuda.set_device(device)
            logger.info(f"使用GPU训练,有效GPU编号:{valid_gpu_ids},主设备:{device}")
            gpu_ids = valid_gpu_ids

    # 数据加载
    sample_ids = collect_sample_ids(base_dir)
    train_ids, val_ids = split_train_val(sample_ids, val_ratio, seed)
    
    train_dataset = SignalDepthDataset(base_dir, train_ids)
    val_dataset = SignalDepthDataset(base_dir, val_ids)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,  
        drop_last=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  
        pin_memory=True,  
        drop_last=False,
        prefetch_factor=1
    )   
    logger.info(f"训练集Loader:{len(train_loader)}个batch,验证集Loader:{len(val_loader)}个batch")

    # 计算OneCycleLR所需的总步数
    train_steps_per_epoch = len(train_loader)  # 每epoch的batch数
    total_training_steps = epochs * train_steps_per_epoch  # 总训练步数

    # 模型、损失函数、优化器初始化
    model = Fnet().to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"已启用多GPU训练,模型包装为DataParallel(设备列表:{gpu_ids})")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 初始化OneCycleLR调度器
    # 计算已完成步数(恢复训练时使用)
    start_epoch = 1
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=lr * 5,  # 学习率峰值
        total_steps=total_training_steps,  # 总训练步数
        pct_start=0.3,  # 前30%步数用于学习率"升温"
        div_factor=25.0,      # 初始lr = max_lr / 25
        final_div_factor=1e4,  # 最小学习率 = max_lr / final_div_factor
        anneal_strategy="cos",  # 后70%步数用余弦退火
        cycle_momentum=True,  # 同时调整动量
        base_momentum=0.85,  # 基础动量
        max_momentum=0.95,    # 最大动量
        last_epoch=-1  
    )
    
    scaler = torch.amp.GradScaler(device=device)
    
    # 初始化早停器
    early_stopper = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        verbose=True
    )
    
    # 模型保存目录初始化
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    best_val_loss = float("inf")
    global_step = 0

    # 从检查点恢复(如果指定)
    if resume_checkpoint and Path(resume_checkpoint).exists():
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, scaler, early_stopper, device
        )
        start_epoch += 1  # 从下一个epoch开始
        if early_stopper.early_stop:
            logger.warning("恢复的早停器已触发停止状态，将重置早停标志以继续训练")
            early_stopper.early_stop = False

    # 初始化TensorBoard
    tb_log_dir = save_dir / "tb_logs"
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard已启动,日志保存路径: {tb_log_dir}")

    # 开始训练
    logger.info("="*50)
    logger.info("开始训练")
    logger.info(f"训练配置:epochs={epochs}, batch_size={batch_size}, max_lr={lr}, "
                f"设备={device}, GPU列表={gpu_ids if len(gpu_ids)>0 else 'CPU'}, "
                f"梯度裁剪阈值={grad_clip}, 早停容忍epoch={early_stop_patience}, "
                f"OneCycleLR总步数={total_training_steps}, 升温比例={0.3}")
    logger.info("="*50)

    for epoch in range(start_epoch, epochs + 1):
        # 训练阶段:传入scheduler(用于每个batch调用step())
        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler, 
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            scaler=scaler,
            grad_clip=grad_clip
        )
        
        torch.cuda.empty_cache()

        # 验证阶段
        val_loss = validate_one_epoch(model, val_loader, device, epoch,gpu_ids)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} - 当前学习率:{current_lr:.6f}")

        # 早停判断
        early_stopper.step(val_loss)
        if early_stopper.early_stop:
            break
        
        # 更新最佳验证损失
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # 记录TensorBoard指标
        writer.add_scalar("Train/Epoch Average Loss", train_loss, global_step=epoch)
        writer.add_scalar("Val/Epoch Average Loss", val_loss, global_step=epoch)
        writer.add_scalar("Learning Rate", current_lr, global_step=epoch)

        # # 记录权重到TensorBoard
        # log_model_params_to_tb(writer, model, step=epoch, log_type="weights")

        # 保存检查点
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopper=early_stopper,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            save_dir=save_dir,
            is_best=is_best
        )
        
        logger.info("-"*50)
    
    # 训练结束
    writer.close()
    logger.info(f"TensorBoard已关闭,日志已完整保存至: {tb_log_dir}")
    
    logger.info("="*50)
    if early_stopper.early_stop:
        logger.info(f"训练因早停机制提前结束！")
    else:
        logger.info(f"训练完成所有{epochs}个epoch!")
    logger.info(f"最终最佳验证损失:{best_val_loss:.4f}")
    logger.info("="*50)


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 配置参数
    BASE_DIR = " "
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4  
    VAL_RATIO = 0.2
    SAVE_DIR = " "
    SEED = 42
    GPU_IDS = [4,5]
    GRAD_CLIP = 0.5
    RESUME_CHECKPOINT = None
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 1e-6

    # 启动训练
    train(
        base_dir=BASE_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        val_ratio=VAL_RATIO,
        save_dir=SAVE_DIR,
        seed=SEED,
        gpu_ids=GPU_IDS,
        grad_clip=GRAD_CLIP,
        resume_checkpoint=RESUME_CHECKPOINT,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA
    )