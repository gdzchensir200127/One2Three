from logging import Logger
import config
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from external.PyTorchEMD.emd import earth_mover_distance

from functions.base import CheckpointRunner
from models.classifier import Classifier
from models.layers.chamfer_wrapper import ChamferDist
from models.p2m import P2MModel
from models.s2p import S2PModel
from models.s2p_4views import S2P_4views_Model
# from models.s2p_2views_test import S2P_2views_test_Model
from models.s2p_4views_Transformer import S2P_4views_Transfomer
from models.s2p_4views_Transformer_add import S2P_4views_Transfomer_add
from models.s2p_4views_proj_v4_ge import Signal2Pixel_Model

# from models.s2p_4views_proj_4view import Signal2Pixel_4views_Model

from models.s2p_4views_proj_4view_test1 import Signal2Pixel_4views_Model

from models.signal_refiner import SignalRefiner
from models.p2s import P2SModel
from models.discriminator import Discriminator
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
# from utils.vis.renderer import MeshRenderer
from utils.vis.displayer import PtDisplayer
import matplotlib.pyplot as plt
import os, json
from fuse.transform import fuse3D
from easydict import EasyDict as edict

class Evaluator(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None, shared_discriminator=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        root = os.path.join(config.DATASET_ROOT, self.options.dataset.dir)
        with open(os.path.join(root, "meta", self.options.dataset.shapenet.shapenet_json), "r") as fp:
            self.json = json.load(fp)
            self.labels_map = sorted(list(self.json.keys()))
        self.labels_map = {i: k for i, k in enumerate(self.labels_map)}
        self.tensor_displayer = None
        
        if self.options.model.name == "pixel2mesh":
            # Renderer for visualization
            # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
            #                              self.options.dataset.mesh_pos)
            # Initialize distance module
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
            # use weighted mean evaluation metrics or not
            self.weighted_mean = self.options.test.weighted_mean
        elif self.options.model.name == 'signal2point':
            self.renderer = None
            self.displayer = True
            # Initialize distance module
            self.weighted_mean = self.options.test.weighted_mean
        elif self.options.model.name == 'signal2point_4views':
            self.renderer = None
            self.displayer = True
            self.weighted_mean = self.options.test.weighted_mean
        elif self.options.model.name == "signal2point_2views":
            self.renderer = None
            self.displayer = True
            self.weighted_mean = self.options.test.weighted_mean    
        elif self.options.model.name == "signal2point_4views_transformer":
            self.renderer = None
            self.displayer = True
            self.weighted_mean = self.options.test.weighted_mean
        elif self.options.model.name == "signal2point_4views_transformer_add":
            self.renderer = None
            self.displayer = True
            self.weighted_mean = self.options.test.weighted_mean
        elif self.options.model.name == "signal2point_GAN":
            self.renderer = None
            self.displayer = True
            self.weighted_mean = self.options.test.weighted_mean
        elif self.options.model.name == "point2signal":
            self.renderer = None
            self.displayer = None
            self.tensor_displayer = True
            self.weighted_mean = self.options.test.weighted_mean
            self.MSE_loss = torch.nn.MSELoss(reduction='mean')
        elif self.options.model.name == "signal2pixel":
            self.renderer = None
            self.displayer = None
            self.depth_displayer = True
            self.weighted_mean = self.options.test.weighted_mean
            self.l1_loss = torch.nn.L1Loss()
            self.renderDepth = 0.747
            self.outViewN = 4
        elif self.options.model.name == "signal2pixel_4views":
            self.renderer = None
            self.displayer = None
            self.depth_displayer = True
            self.weighted_mean = self.options.test.weighted_mean
            self.l1_loss = torch.nn.L1Loss()
            self.renderDepth = 0.747
            self.outViewN = 4
        elif self.options.model.name == "signal2pixel_4views_out4":
            self.renderer = None
            self.displayer = None
            self.depth_displayer = True
            self.weighted_mean = self.options.test.weighted_mean
            self.l1_loss = torch.nn.L1Loss()
            self.renderDepth = 0.747
            self.outViewN = 4
        elif self.options.model.name == "signal_refiner":
            self.renderer = None
            self.displayer = None
            self.depth_displayer = None
            self.tensor_displayer = True
        else:
            self.renderer = None
        self.num_classes = self.options.dataset.num_classes

        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
            elif self.options.model.name == "signal2point":
                self.model = S2PModel(self.options.model)
            elif self.options.model.name == "signal2point_4views":
                self.model = S2P_4views_Model(self.options.model)
            # elif self.options.model.name == "signal2point_2views":
            #     self.model = S2P_2views_test_Model(self.options.model)
            elif self.options.model.name == "signal2point_4views_transformer":
                self.model = S2P_4views_Transfomer(self.options.model)
            elif self.options.model.name == "signal2point_4views_transformer_add":
                self.model = S2P_4views_Transfomer_add(self.options.model)
            elif self.options.model.name == "signal2point_GAN":
                self.model = S2P_4views_Model(self.options.model)
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            elif self.options.model.name == "point2signal":
                self.model = P2SModel(self.options.model)
            elif self.options.model.name == "signal2pixel":
                self.model = Signal2Pixel_Model(self.options.model)
            elif self.options.model.name == "signal2pixel_4views":
                self.model = Signal2Pixel_4views_Model(self.options.model)         
            elif self.options.model.name == 'signal2pixel_4views_out4':
                self.model = Signal2Pixel_4views_Model(self.options.model) 
            elif self.options.model.name == 'signal_refiner':
                self.model = SignalRefiner()
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            if self.options.model.name == 'signal2point_GAN':
                self.Discriminator = torch.nn.DataParallel(self.Discriminator, device_ids=self.gpus).cuda()

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0
        if self.options.model.name == "signal2pixel" or self.options.model.name == "signal2pixel_4views" or self.options.model.name == 'signal2pixel_4views_out4':
            self.cfg = edict()
            self.cfg.device = next(self.model.parameters()).device
            self.cfg.batchSize = self.options.test.batch_size
            self.cfg.path = root
            self.cfg.sampleN = 100
            self.cfg.renderDepth = 0.747
            self.cfg.BNepsilon = 1e-5
            self.cfg.BNdecay = 0.999
            self.cfg.scale = 1
            # self.cfg.inputViewN = 24
            # ------ below automatically set ------
            self.cfg.outH, self.cfg.outW = 128, 128
            self.cfg.outViewN = 4
            self.cfg.H, self.cfg.W = 128, 128
            self.cfg.Khom3Dto2D = torch.Tensor([[self.cfg.W, 0, 0, self.cfg.W / 2],
                                        [0, -self.cfg.H, 0, self.cfg.H / 2],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]]).float().to(self.cfg.device)
            self.cfg.Khom2Dto3D = torch.Tensor([[self.cfg.outW  / self.cfg.scale, 0, 0, self.cfg.outW / 2 / self.cfg.scale],
                                        [0, -self.cfg.outH / self.cfg.scale, 0, self.cfg.outH / 2/ self.cfg.scale],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]]).float().to(self.cfg.device)
            self.cfg.fuseTrans_q = torch.from_numpy(
                    np.load(f"{self.cfg.path}/trans_fuse{self.cfg.outViewN}_q.npy")).to(self.cfg.device)
            # self.cfg.fuseTrans_t = torch.from_numpy(
            #         np.load(f"{self.cfg.path}/trans_fuse{self.cfg.outViewN}_t.npy")).to(self.cfg.device)
            self.cfg.fuseTrans_t  = torch.Tensor([0, 0, -self.cfg.renderDepth]) \
                  .repeat([self.cfg.outViewN, 1]).to(self.cfg.device) # [V,3]

    def models_dict(self):
        return {'model': self.model}

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def random_downsample(self, input_tensor: torch.Tensor, m: int) -> torch.Tensor:
        """
        对形状为 [1, n, 3] 的点云张量进行随机降采样,保留 m 个点。
        
        Args:
            input_tensor (torch.Tensor): 输入点云张量,形状为 [1, n, 3]。
            m (int): 降采样后的目标点数（需满足 m <= n）。
        
        Returns:
            torch.Tensor: 降采样后的张量,形状为 [1, m, 3]。
        """
        assert input_tensor.dim() == 3 and input_tensor.shape[0] == 1, "输入张量形状应为 [1, n, 3]"
        n = input_tensor.shape[1]
        if m > n:
            raise ValueError(f"m ({m}) 不能超过原始点数 n ({n})")
        
        # 生成随机索引（与输入张量在同一设备上）
        device = input_tensor.device
        indices = torch.randperm(n, device=device)[:m]
        
        # 按索引提取点
        downsampled_tensor = input_tensor[:, indices, :]
        return downsampled_tensor

    def evaluate_emd(self, pred_vertices, gt_points, labels):
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            points_a = pred_vertices[i].unsqueeze(0)
            points_b = gt_points[i].unsqueeze(0)
            n = points_a.size(1)
            m = points_b.size(1)
            if n>m:
                points_a = self.random_downsample(points_a, m)
            elif n<m:
                points_b = self.random_downsample(points_b, n)
            d = earth_mover_distance(points_a, points_b, transpose=True)
            d = d.cpu().numpy()
            self.emd[label].update(d)

    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, labels):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            points_a = pred_vertices[i].unsqueeze(0)
            points_b = gt_points[i].unsqueeze(0)
            fun = ChamferDist()
            d1,d2,_,_ = fun(points_a, points_b)
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  
            d1 = np.sqrt(d1)
            d2 = np.sqrt(d2) # 平方距离转换成距离
            self.chamfer_distance[label].update(np.mean(d1) + np.mean(d2))
            self.f1_tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-2))
            self.f1_2tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 4E-2))
    
    def evaluate_signal_similarity(self, pred_I, pred_Q, gt_I, gt_Q, labels):
        batch_size = pred_I.size(0)
        for i in range(batch_size):
            label = labels[i].cpu().item()
            I_MSE = F.mse_loss(pred_I[i], gt_I[i], reduction="mean")
            Q_MSE = F.mse_loss(pred_Q[i], gt_Q[i], reduction="mean")
            MSE = I_MSE * 0.5 + Q_MSE * 0.5
            I_similarity = F.cosine_similarity(pred_I[i][pred_I.size(1)//2], gt_I[i][gt_I.size(1)//2], dim=0).cpu().item()
            Q_similarity = F.cosine_similarity(pred_Q[i][pred_Q.size(1)//2], gt_Q[i][gt_Q.size(1)//2], dim=0).cpu().item()
            similarity = I_similarity*0.5 + Q_similarity*0.5
            self.I_similarity[label].update(I_similarity)
            self.Q_similarity[label].update(Q_similarity)
            self.similarity[label].update(similarity)
            self.I_MSE[label].update(I_MSE)
            self.Q_MSE[label].update(Q_MSE)
            self.MSE[label].update(MSE)
    
    def evaluate_depth_l1_dist(self, depth, depthGT, mask, labels):
        batch_size = depth.size(0)
        for i in range(batch_size):
            label = labels[i].cpu().item()
            depth[i][~mask[i]] = self.renderDepth
            depth_dist = self.l1_loss(depth[i], depthGT[i])
            self.depth_l1dist[label].update(depth_dist)

    def evaluate_accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        top_k = [1, 5]
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1.0 / batch_size)
            if k == 1:
                self.acc_1.update(acc)
            elif k == 5:
                self.acc_5.update(acc)
    def evaluate_signal_refine_accuracy(self, output, inputBatch):
        signal_pred = output
        signal_simulate = inputBatch["simulate_signal_abs"]
        signal_gt = inputBatch['GT_signal_abs']
        # mask_true = inputBatch['GT_signal_abs_mask']
        # mask_binary = (mask_pred > 0.5).float()  # 二值化
        
        def pearson_correlation_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """
            计算两个张量的皮尔逊相关系数,并返回全局均值标量
            输入形状: [BatchSize, 2800, 9]
            输出形状: 标量
            """
            a = minmax_along_time(a)
            b = minmax_along_time(b)
            # 沿时间维度（dim=1）计算均值
            a_mean = torch.mean(a, dim=1, keepdim=True)
            b_mean = torch.mean(b, dim=1, keepdim=True)
            
            # 中心化
            a_centered = a - a_mean
            b_centered = b - b_mean
            
            # 计算协方差
            covariance = torch.sum(a_centered * b_centered, dim=1)  # [BatchSize, 9]
            
            # 计算标准差（使用有偏估计,与协方差分母一致）
            a_std = torch.std(a, dim=1, unbiased=False)  # [BatchSize, 9]
            b_std = torch.std(b, dim=1, unbiased=False)  # [BatchSize, 9]
            
            # 计算相关系数（避免除以零）
            eps = 1e-8
            corr = covariance / (a_std * b_std + eps)
            
            # 全局均值（跨 Batch 和通道）
            return torch.mean(corr)
        
        def cosine_similarity_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """
            计算两个张量的余弦相似度,并返回全局均值标量
            输入形状: [BatchSize, 2800, 9]
            输出形状: 标量
            """
            # 计算每个样本和通道的余弦相似度
            cos_sim = F.cosine_similarity(a, b, dim=1)  # [BatchSize, 9]
            
            # 全局均值（跨 Batch 和通道）
            return torch.mean(cos_sim)
        
        def minmax_along_time(tensor: torch.Tensor, eps: float = 1e-18) -> torch.Tensor:
            """
            沿时间维度（dim=1）归一化到 [0, 1]
            输入形状: [BatchSize, 2800, 9]
            输出形状: [BatchSize, 2800, 9]
            """
            min_val = torch.min(tensor, dim=1, keepdim=True)[0]  # [BatchSize, 1, 9]
            max_val = torch.max(tensor, dim=1, keepdim=True)[0]  # [BatchSize, 1, 9]
            
            # 归一化
            normalized = (tensor - min_val) / (max_val - min_val + eps)
            return normalized

        self.cos_similarity_pred_gt.update(cosine_similarity_scalar(signal_pred, signal_gt))
        self.cos_similarity_simulate_gt.update(cosine_similarity_scalar(signal_simulate, signal_gt))
        self.pearson_correlation_pred_gt.update(pearson_correlation_scalar(signal_pred, signal_gt))
        self.pearson_correlation_simulate_gt.update(pearson_correlation_scalar(signal_simulate, signal_gt))
        self.mse_pred_gt.update(F.mse_loss(minmax_along_time(signal_pred) , minmax_along_time(signal_gt)))
        self.mse_simulate_gt.update(F.mse_loss(minmax_along_time(signal_simulate) , minmax_along_time(signal_gt)))
        self.mae_pred_gt.update(F.l1_loss(minmax_along_time(signal_pred) , minmax_along_time(signal_gt)))
        self.mae_simulate_gt.update(F.l1_loss(minmax_along_time(signal_simulate) , minmax_along_time(signal_gt)))


    def evaluate_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            if self.options.model.name == "signal2point":
            # Get ground truth
                signal_real = input_batch["signal_real"]
                signal_imag = input_batch["signal_imag"]
                # predict with model
                out = self.model(signal_real,signal_imag)
            elif self.options.model.name == "signal2point_4views":
                signals = input_batch["signals"]
                out = self.model(signals)
            elif self.options.model.name == "signal2point_2views":
                signals = input_batch["signals"]
                out = self.model(signals)
            elif self.options.model.name == "signal2point_4views_transformer":
                signals = input_batch["signals"]
                out = self.model(signals)
            elif self.options.model.name == "signal2point_4views_transformer_add":
                signals = input_batch["signals"]
                out = self.model(signals)
            elif self.options.model.name == "signal2point_GAN":
                signals = input_batch["signals"]
                out = self.model(signals)
            elif self.options.model.name == "signal2pixel":
                signals = input_batch["signals"]
                out = []
                for i in range(4):
                    out_i = self.model(signals[i])
                    out.append(out_i)
            elif self.options.model.name == "signal2pixel_4views":
                signals = input_batch["signals"]
                out = []
                for i in range(4):
                    out_i = self.model(signals)
                    out.append(out_i)
                    signals.append(signals.pop(0))
            elif self.options.model.name == "signal2pixel_4views_out4":
                signals = input_batch["signals"]
                
                if self.evaluate_step_count % 1000 == 0 and self.evaluate_step_count != 0:
                    save_path = f'/home/zhang_muxin/Signal2PC/datasets/data/N32/xianshiqi_features/step_{self.evaluate_step_count}'
                    out = self.model(signals, save_path=save_path)
                else:
                    out = self.model(signals)
                    
            elif self.options.model.name == "point2signal":
                out = self.model(input_batch["points"], input_batch["start_bin"], input_batch["end_bin"])
            elif self.options.model.name == "signal_refiner":
                out = self.model(input_batch["simulate_signal_abs"])
            if self.options.model.name == "pixel2mesh":
                pred_vertices = out["pred_coord"][-1]
                gt_points = input_batch["points_orig"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "signal2point":
                pred_vertices = out["pred_coord"]
                gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "signal2point_4views":
                pred_vertices = out["pred_coord"]
                gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "signal2point_2views":
                pred_vertices = out["pred_coord"]
                gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "signal2point_4views_transformer":
                pred_vertices = out["pred_coord"]
                gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "signal2point_4views_transformer_add":
                pred_vertices = out["pred_coord"]
                gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "signal2point_GAN":
                pred_vertices = out["pred_coord"]
                gt_points = input_batch["points"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "point2signal":
                pred_I = out["I"]
                pred_Q = out["Q"]
                gt_I = input_batch["signal_real"][:,:,::self.options.model.downsample_factor] # 下采样到 8 * 9 * 280
                gt_Q = input_batch["signal_imag"][:,:,::self.options.model.downsample_factor]
                if isinstance(gt_I, list):
                    gt_I = [I.cuda() for I in gt_I]
                if isinstance(gt_Q, list):
                    gt_Q = [Q.cuda() for Q in gt_Q]
                self.evaluate_signal_similarity(pred_I, pred_Q, gt_I, gt_Q, input_batch["labels"])
            elif self.options.model.name == "signal2pixel":
                for i in range(4):
                    depthGT = input_batch["project"][:,i,:,:].unsqueeze(1)
                    maskGT = depthGT != 0
                    depthGT[~maskGT] = self.renderDepth
                    XYZ, maskLogit = out[i][0], out[i][1]
                    depth = XYZ[:,  2: 3, :,  :]
                    mask = (maskLogit > 0)
                    self.evaluate_depth_l1_dist(depth, depthGT, mask, input_batch["labels"])
            elif self.options.model.name == "signal2pixel_4views":
                for i in range(4):
                    depthGT = input_batch["project"][:,i,:,:].unsqueeze(1)
                    maskGT = depthGT != 0
                    depthGT[~maskGT] = self.renderDepth
                    XYZ, maskLogit = out[i][0], out[i][1]
                    depth = XYZ[:,  2: 3, :,  :]
                    mask = (maskLogit > 0)
                    self.evaluate_depth_l1_dist(depth, depthGT, mask, input_batch["labels"])
            elif self.options.model.name == 'signal2pixel_4views_out4':
                depthGT = input_batch["project"]
                maskGT = depthGT != 0
                depthGT[~maskGT] = self.renderDepth
                depth, maskLogit = out[0], out[1]
                # depth = XYZ[:,2*self.outViewN: 3*self.outViewN, :, :]
                mask = (maskLogit > 0)
                self.evaluate_depth_l1_dist(depth, depthGT, maskGT, input_batch["labels"])
                
                XYZid, ML = self.depth2point(out[0])
                XYZid_GT, ML_GT = self.depth2point(input_batch["project"])
                pc = input_batch["points"]
                for batch in range(XYZid.shape[0]):
                    XYZid_b = XYZid[batch,:,:]
                    ML_b = ML[batch,:,:]
                    ML_b = ML_b.expand_as(XYZid_b)
                    XYZid_b = XYZid_b[ML_b]
                    XYZid_b = XYZid_b.view(1, 3, -1)
                    XYZid_b = XYZid_b.transpose(1, 2)
                    XYZid_b = self.pc_rotation_scale(XYZid_b, 90, 0.5) # 为了让深度图中物体占据更多像素,在渲染深度图时对模型进行了两倍放缩,现在恢复真实大小

                    XYZid_b_GT = XYZid_GT[batch,:,:]
                    ML_b_GT = ML_GT[batch,:,:]
                    ML_b_GT = ML_b_GT.expand_as(XYZid_b_GT)
                    XYZid_b_GT = XYZid_b_GT[ML_b_GT]
                    XYZid_b_GT = XYZid_b_GT.view(1, 3, -1)
                    XYZid_b_GT = XYZid_b_GT.transpose(1, 2)
                    XYZid_b_GT = self.pc_rotation_scale(XYZid_b_GT, 90, 0.5)
                    
                    pc_instance = pc[batch,:,:].view(1,3,-1)
                    pc_instance = pc_instance.transpose(1, 2)
                    pc_instance = self.pc_rotation_scale(pc_instance, 0, 0.5)
                    self.evaluate_chamfer_and_f1(XYZid_b, pc_instance, labels=input_batch["labels"][batch:batch+1])
                    self.evaluate_emd(XYZid_b, pc_instance, labels=input_batch["labels"][batch:batch+1])

            elif self.options.model.name == "classifier":
                self.evaluate_accuracy(out, input_batch["labels"])
            elif self.options.model.name == "signal_refiner":
                self.evaluate_signal_refine_accuracy(out, input_batch)
                
        return out

    # noinspection PyAttributeOutsideInit
    def evaluate(self):
        self.logger.info("Running evaluations...")

        # clear evaluate_step_count, but keep total count uncleared
        self.evaluate_step_count = 0

        test_data_loader = DataLoader(self.dataset,
                                      batch_size=self.options.test.batch_size * self.options.num_gpus,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.test.shuffle,
                                      collate_fn=self.dataset_collate_fn)

        if self.options.model.name == "pixel2mesh":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        
        elif self.options.model.name == "signal2point":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        
        elif self.options.model.name == "signal2point_4views":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]

        elif self.options.model.name == "signal2point_2views":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]

        elif self.options.model.name == "signal2point_4views_transformer":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        
        elif self.options.model.name == "signal2point_4views_transformer_add":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        
        elif self.options.model.name == "signal2point_GAN":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]

        elif self.options.model.name == "signal2pixel":
            self.depth_l1dist = [AverageMeter() for _ in range(self.num_classes)]
        
        elif self.options.model.name == "signal2pixel_4views":
            self.depth_l1dist = [AverageMeter() for _ in range(self.num_classes)]
        
        elif self.options.model.name == "signal2pixel_4views_out4":
            self.depth_l1dist = [AverageMeter() for _ in range(self.num_classes)]
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
            self.emd = [AverageMeter() for _ in range(self.num_classes)]

        elif self.options.model.name == "point2signal":

            self.I_similarity = [AverageMeter() for _ in range(self.num_classes)]
            self.Q_similarity = [AverageMeter() for _ in range(self.num_classes)]
            self.similarity = [AverageMeter() for _ in range(self.num_classes)]
            self.I_MSE = [AverageMeter() for _ in range(self.num_classes)]
            self.Q_MSE = [AverageMeter() for _ in range(self.num_classes)]
            self.MSE = [AverageMeter() for _ in range(self.num_classes)]
        elif self.options.model.name == "signal_refiner":
            self.recon_loss = AverageMeter()
            self.cos_similarity_simulate_gt = AverageMeter()
            self.cos_similarity_pred_gt = AverageMeter()
            self.pearson_correlation_simulate_gt = AverageMeter()
            self.pearson_correlation_pred_gt = AverageMeter()
            self.mse_simulate_gt = AverageMeter()
            self.mse_pred_gt = AverageMeter()
            self.mae_simulate_gt = AverageMeter()
            self.mae_pred_gt = AverageMeter()
            # self.mask_loss = AverageMeter()

        elif self.options.model.name == "classifier":
            self.acc_1 = AverageMeter()
            self.acc_5 = AverageMeter()

        # Iterate over all batches in an epoch
        for step, batch in enumerate(test_data_loader):
            # Send input to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Run evaluation step
            out = self.evaluate_step(batch)

            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, out)

            # add later to log at step 0
            self.evaluate_step_count += 1
            self.total_step_count += 1

        for key, val in self.get_result_summary().items():
            scalar = val
            if isinstance(val, AverageMeter):
                scalar = val.avg
            self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))
            self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)

    def average_of_average_meters(self, average_meters):
        # average_meters = [meter for meter in average_meters if meter.avg != 0]
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        
        updated_meters = [meter for meter in average_meters if meter.count > 0]
        avg = sum([meter.avg for meter in updated_meters]) / len(updated_meters) if len(updated_meters) > 0 else 0.
        # avg = sum([meter.avg for meter in average_meters]) / len(average_meters)
        
        ret = AverageMeter()
        if self.weighted_mean:
            ret.val, ret.avg = avg, weighted_avg
        else:
            ret.val, ret.avg = weighted_avg, avg
        return ret

    def get_result_summary(self):
        if self.options.model.name == "pixel2mesh":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "signal2point":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "signal2point_4views":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "signal2point_2views":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "signal2point_4views_transformer":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "signal2point_4views_transformer_add":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "signal2point_GAN":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "classifier":
            return {
                "acc_1": self.acc_1,
                "acc_5": self.acc_5,
            }
        elif self.options.model.name == "point2signal":
            return {
                "I_similarity": self.average_of_average_meters(self.I_similarity),
                "Q_similarity": self.average_of_average_meters(self.Q_similarity),
                "similarity": self.average_of_average_meters(self.similarity),
                "I_MSE": self.average_of_average_meters(self.I_MSE),
                "Q_MSE": self.average_of_average_meters(self.Q_MSE),
                "MSE": self.average_of_average_meters(self.MSE)
            }
        elif self.options.model.name == 'signal_refiner':
            return {
                "cos_similarity_pred_gt":self.cos_similarity_pred_gt,
                "cos_similarity_simulate_gt":self.cos_similarity_simulate_gt,
                "pearson_correlation_pred_gt":self.pearson_correlation_pred_gt,
                "pearson_correlation_simulate_gt":self.pearson_correlation_simulate_gt,
                "mse_simulate_gt":self.mse_simulate_gt,
                "mse_pred_gt":self.mse_pred_gt,
                "mae_simulate_gt":self.mae_simulate_gt,
                "mae_pred_gt":self.mae_pred_gt,
                # "mask": self.mask_loss
            }
        elif self.options.model.name == "signal2pixel":
            ret = {
                "depth_dist": self.average_of_average_meters(self.depth_l1dist)
            }
            for i, avgmeter in enumerate(self.depth_l1dist):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_depth_dist"] = avgmeter
            return ret
        elif self.options.model.name == "signal2pixel_4views":
            ret = {
                "depth_dist": self.average_of_average_meters(self.depth_l1dist)
            }
            for i, avgmeter in enumerate(self.depth_l1dist):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_depth_dist"] = avgmeter
            return ret
        elif self.options.model.name == "signal2pixel_4views_out4":
            ret = {
                "depth_dist": self.average_of_average_meters(self.depth_l1dist),
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
                "emd": self.average_of_average_meters(self.emd)
            }
            for i, avgmeter in enumerate(self.depth_l1dist):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_depth_dist"] = avgmeter
            for i, avgmeter in enumerate(self.chamfer_distance):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_cd"] = avgmeter
            for i, avgmeter in enumerate(self.f1_tau):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_f1_tau"] = avgmeter
            for i, avgmeter in enumerate(self.f1_2tau):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_f1_2tau"] = avgmeter
            for i, avgmeter in enumerate(self.emd):
                if avgmeter.avg != 0:
                    ret[self.labels_map[i] + "_emd"] = avgmeter
            return ret
    def plot_signal(self, signals: torch.Tensor):
        '''
        signals: [9*280]
        '''
        signals = np.squeeze(signals.cpu().numpy())
        x = np.linspace(0,10,9)
        y = np.linspace(0,10,280)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(signals.shape[0]):
            ax.plot(np.array([x[i] for _ in range(signals.shape[1])]),y,signals[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig

    def render_heatmaps(self, magnitude, rows=None, cols=None):
        """
        将幅值张量渲染为热度图,并拼接成一张大图。

        参数:
            magnitude (torch.Tensor): 幅值张量,形状为 (batch_size, height, width)。
            rows (int): 子图的行数。如果为 None,则自动计算。
            cols (int): 子图的列数。如果为 None,则自动计算。

        返回:
            image (np.ndarray): 渲染后的图像,形状为 (H, W, 3)。
        """
        batch_size, height, width = magnitude.shape

        # 如果没有指定行数和列数,则自动计算
        if rows is None or cols is None:
            cols = int(np.ceil(np.sqrt(batch_size)))  # 列数为 batch_size 的平方根向上取整
            rows = int(np.ceil(batch_size / cols))    # 行数为 batch_size 除以列数向上取整

        # 创建画布
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # 每个子图大小为 4x4
        fig.suptitle("Signal Magnitude Heatmaps for Batch")

        # 如果 batch_size 小于 rows * cols,去掉多余的空子图
        if batch_size < rows * cols:
            for i in range(batch_size, rows * cols):
                fig.delaxes(axes.flatten()[i])

        # 遍历 batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的幅值
            sample_magnitude = magnitude[i].numpy()

            # 计算子图的行和列索引
            row = i // cols
            col = i % cols

            # 在当前子图中绘制热度图
            ax = axes[row, col]
            im = ax.imshow(sample_magnitude, cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f'Sample {i}')
            fig.colorbar(im, ax=ax)  # 为每个子图添加颜色条

        # 调整布局
        plt.tight_layout()

        # 将 Matplotlib 图像转换为 NumPy 数组
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 转换为 (H, W, C)

        # 关闭 Matplotlib 图像
        plt.close(fig)

        return image



    def evaluate_summaries(self, input_batch, out_summary):
        self.logger.info("Test Step %06d/%06d (%06d) " % (self.evaluate_step_count,
                                                          len(self.dataset) // (
                                                                  self.options.num_gpus * self.options.test.batch_size),
                                                          self.total_step_count,) \
                         + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                                      for key, val in self.get_result_summary().items()]))
        if "labels" in input_batch:
            self.summary_writer.add_histogram("eval_labels", input_batch["labels"].cpu().numpy(),
                                            self.total_step_count)
        if self.displayer is not None:
            # Do visualization for the first 2 images of the batch
            # display_pt = self.displayer.s2p_batch_visualize(input_batch, out_summary)
            
            self.summary_writer.add_mesh(tag='eval_gt_pc', vertices = input_batch['points'][0:1,:,:],global_step=self.total_step_count)
            self.summary_writer.add_mesh(tag='eval_pred_pc', vertices = out_summary['pred_coord'][0:1,:,:],global_step=self.total_step_count)

            # self.summary_writer.add_image("eval_points", display_pt, self.total_step_count)
            self.summary_writer.add_histogram("length_distribution", input_batch["length"].cpu().numpy(),
                                              self.total_step_count)
        if self.tensor_displayer is not None:
            if self.options.model.name == 'signal_refiner':
                GT_magnitude = input_batch['GT_signal_abs'].cpu()
                GT_img = self.render_heatmaps(GT_magnitude)
                self.summary_writer.add_image('GT Signal Magnitude Heatmaps', GT_img, global_step=self.total_step_count, dataformats='HWC')
                pre_magnitude = out_summary
                # pre_magnitude = pre_magnitude * (pre_mask > 0.5).float()
                pre_img = self.render_heatmaps(pre_magnitude.cpu())
                self.summary_writer.add_image('predict Signal Magnitude Heatmaps', pre_img, global_step=self.total_step_count, dataformats='HWC')
                simulate_magnitude = input_batch['simulate_signal_abs'] .cpu()
                simulate_img = self.render_heatmaps(simulate_magnitude)
                self.summary_writer.add_image('Simulate Signal Magnitude Heatmaps', simulate_img, global_step=self.total_step_count, dataformats='HWC')
            else:
                self.summary_writer.add_figure("gt_I",self.plot_signal(input_batch["signal_real"][0:1,:,::self.options.model.downsample_factor]),global_step=self.total_step_count)
                self.summary_writer.add_figure("pred_I",self.plot_signal(out_summary["I"][0:1,:,:]),global_step=self.total_step_count)
                self.summary_writer.add_figure("gt_Q",self.plot_signal(input_batch["signal_imag"][0:1,:,::self.options.model.downsample_factor]),global_step=self.total_step_count)
                self.summary_writer.add_figure("pred_Q",self.plot_signal(out_summary["Q"][0:1,:,:]),global_step=self.total_step_count)
            
        if self.depth_displayer is not None:
            if self.options.model.name == "signal2pixel_4views":
                final_images = []
                for i in range(4):
                    depthGT = input_batch["project"][:,i,:,:].unsqueeze(1)
                    # maskGT = depthGT != 0
                    # depthGT[~maskGT] = self.renderDepth
                    XYZ, maskLogit = out_summary[i][0], out_summary[i][1]
                    mask = (maskLogit > 0)
                    depth = XYZ[:, 2: 3, :,  :]
                    # depth[~mask] = self.renderDepth
                    depth = depth.cpu().numpy()
                    depthGT = depthGT.cpu().numpy()
                    batch_size = depth.shape[0]
                    depth_image = []
                    for j in range(batch_size):
                        depth_image.append(np.squeeze(depth[j]))
                    depth_image = np.concatenate(depth_image, axis=1)
                    depthGT_image = []
                    for j in range(batch_size):
                        depthGT_image.append(np.squeeze(depthGT[j]))
                    depthGT_image = np.concatenate(depthGT_image, axis=1)
                    images = [depthGT_image, depth_image]
                    images = np.concatenate(images, axis=0)
                    final_images.append(images)
                final_images = np.concatenate(final_images, axis=0)
                self.summary_writer.add_image("depth", final_images, dataformats='HW',global_step=self.total_step_count)
                # gt 3D fuse
                XYZ = []
                XGT, YGT = torch.meshgrid([
                    torch.arange(self.cfg.outH), # [H,W]
                    torch.arange(self.cfg.outW)]) # [H,W]
                XGT, YGT = XGT.float(), YGT.float()
                XYGT = torch.cat([
                    XGT.repeat([self.outViewN, 1, 1]), 
                    YGT.repeat([self.outViewN, 1, 1])], dim=0) #[2V,H,W]
                XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) # [1,2V,H,W] 
                XYGT = XYGT.repeat([self.cfg.batchSize,1,1,1])
                XYZ.append(XYGT)
                maskLogits = []
                for i in range(4):
                    depthGT = input_batch["project"][:,i,:,:].unsqueeze(1).permute(0,1,3,2)
                    maskGT = depthGT != self.renderDepth
                    XYZ.append(depthGT)
                    maskLogits.append(maskGT)
                XYZ = torch.concat(XYZ, dim=1)
                maskLogits = torch.concat(maskLogits, dim=1)
                XYZid, ML = fuse3D(
                        self.cfg, XYZ, maskLogits, self.cfg.fuseTrans_q, self.cfg.fuseTrans_t) # [B,3,VHW],[B,1,VHW]
                ML = ML > 0
                XYZid = XYZid[0,:,:]
                ML = ML[0,:,:]
                ML = ML.expand_as(XYZid)
                XYZid = XYZid[ML]
                XYZid = XYZid.view(1, 3, -1)
                XYZid = XYZid.transpose(1, 2)
                self.summary_writer.add_mesh(tag='eval_gt_pc', vertices = XYZid, global_step=self.total_step_count)

                # 3D fuse
                X = []
                Y = []
                depth = []
                maskLogits = []
                for i in range(4):
                    XYZ, maskLogit = out_summary[i][0], out_summary[i][1]
                    X.append(XYZ[:, :1, :, :])
                    Y.append(XYZ[:, 1 : 2, :, :])
                    depth.append(XYZ[:, 2 : 3, :, :].permute(0,1,3,2))
                    maskLogits.append(maskLogit.permute(0,1,3,2))
                
                X = torch.concat(X, dim=1)
                Y = torch.concat(Y, dim=1)
                depth = torch.concat(depth, dim=1)
                XYZ = torch.concat([X,Y,depth], dim=1)
                maskLogits = torch.concat(maskLogits, dim=1)
                XYZid, ML = fuse3D(
                        self.cfg, XYZ, maskLogits, self.cfg.fuseTrans_q, self.cfg.fuseTrans_t) # [B,3,VHW],[B,1,VHW]
                ML = ML > 0
                XYZid = XYZid[0,:,:]
                ML = ML[0,:,:]
                ML = ML.expand_as(XYZid)
                XYZid = XYZid[ML]
                XYZid = XYZid.view(1, 3, -1)
                XYZid = XYZid.transpose(1, 2)
                self.summary_writer.add_mesh(tag='eval_pred_pc', vertices = XYZid, global_step=self.total_step_count)

                # for i in range(4):
                #     X = []
                #     Y = []
                #     depth = []
                #     maskLogits = []
                #     XYZ, maskLogit = out_summary[i][0], out_summary[i][1]
                #     X.append(XYZ[:, :1, :, :])
                #     Y.append(XYZ[:, 1 : 2, :, :])
                #     depth.append(XYZ[:, 2 : 3, :, :].permute(0,1,3,2))
                #     maskLogits.append(maskLogit.permute(0,1,3,2))

                #     X = torch.concat(X, dim=1)
                #     Y = torch.concat(Y, dim=1)
                #     depth = torch.concat(depth, dim=1)
                #     XYZ = torch.concat([X,Y,depth], dim=1)
                #     maskLogits = torch.concat(maskLogits, dim=1)
                #     self.cfg.outViewN = 1
                #     XYZid, ML = fuse3D(
                #             self.cfg, XYZ, maskLogits, self.cfg.fuseTrans_q[i:i+1,:], self.cfg.fuseTrans_t[i:i+1,:]) # [B,3,VHW],[B,1,VHW]
                #     self.cfg.outViewN = 4
                #     ML = ML > 0
                #     XYZid = XYZid[0,:,:]
                #     ML = ML[0,:,:]
                #     ML = ML.expand_as(XYZid)
                #     XYZid = XYZid[ML]
                #     XYZid = XYZid.view(1, 3, -1)
                #     XYZid = XYZid.transpose(1, 2)
                #     self.summary_writer.add_mesh(tag='eval_pred_pc'+str(i), vertices = XYZid, global_step=self.total_step_count)

            elif self.options.model.name == 'signal2pixel_4views_out4':
                self.logger.info("options.model.name == 'signal2pixel_4views_out4'")
                # depth image
                final_images = []
                XYZ, maskLogit = out_summary[0], out_summary[1]
                batch_size = XYZ.shape[0]
                self.logger.info(f"XYZ batch_size: {batch_size}, cfg.batchSize: {self.cfg.batchSize}")
                # if batch_size != self.cfg.batchSize:
                #     return
                for i in range(4):
                    self.logger.info(f"input_batch keys: {input_batch.keys()}")
                    depthGT = input_batch["project"][:,i,:,:].unsqueeze(1)
                    mask = (maskLogit > 0)
                    depth = XYZ[:, i:i+1, :,  :]
                    depth = depth.cpu().numpy()
                    depth[depth==self.renderDepth] = 0
                    depthGT = depthGT.cpu().numpy()
                    depthGT[depthGT==self.renderDepth] = 0
                    depth_image = []
                    for j in range(batch_size):
                        depth_image.append(np.squeeze(depth[j]))
                        self.summary_writer.add_image("depth_pre", np.squeeze(depth[j]), dataformats='HW', global_step=(self.total_step_count * 32)+(j*4)+i)
                    depth_image = np.concatenate(depth_image, axis=1)
                    depthGT_image = []
                    for j in range(batch_size):
                        depthGT_image.append(np.squeeze(depthGT[j]))
                        self.summary_writer.add_image("depth_gt", np.squeeze(depthGT[j]), dataformats='HW', global_step=(self.total_step_count * 32)+(j*4)+i)
                    depthGT_image = np.concatenate(depthGT_image, axis=1)
                    images = [depthGT_image, depth_image]
                    images = np.concatenate(images, axis=0)
                    final_images.append(images)
                final_images = np.concatenate(final_images, axis=0)
                self.summary_writer.add_image("depth", final_images, dataformats='HW',global_step=self.total_step_count)
                
                # gt 3D pointcloud
                self.logger.info("gt 3D pointcloud")
                XYZid, ML = self.depth2point(input_batch["project"])
                for batch in range(XYZid.shape[0]):
                    XYZid_b = XYZid[batch,:,:]
                    ML_b = ML[batch,:,:]
                    ML_b = ML_b.expand_as(XYZid_b)
                    XYZid_b = XYZid_b[ML_b]
                    XYZid_b = XYZid_b.view(1, 3, -1)
                    XYZid_b = XYZid_b.transpose(1, 2)
                    XYZid_b = self.pc_rotation_scale(XYZid_b, 90)
                    self.summary_writer.add_mesh(tag='eval_gt_pc', vertices = XYZid_b, global_step=self.total_step_count*self.cfg.batchSize+batch)

                    points = XYZid_b.squeeze(0).cpu().numpy()
                    # self.save_ply(points, os.path.join(input_batch["save_path"][batch], str(self.total_step_count*8+batch)+'_gt.ply'))
                    self.save_ply(points, os.path.join(input_batch["save_path"][batch], str(batch+1)+'_gt.ply'))
                    self.logger.info("Saving gt ply files...")
                    
                # predict 3D pointcloud
                XYZid, ML = self.depth2point(out_summary[0])
                for batch in range(XYZid.shape[0]):
                    self.logger.info("Now batch =")
                    self.logger.info(batch)
                    self.logger.info("total_step_count =")
                    self.logger.info(self.total_step_count)
                    XYZid_b = XYZid[batch,:,:]
                    ML_b = ML[batch,:,:]
                    ML_b = ML_b.expand_as(XYZid_b)
                    XYZid_b = XYZid_b[ML_b]
                    XYZid_b = XYZid_b.view(1, 3, -1)
                    XYZid_b = XYZid_b.transpose(1, 2)
                    XYZid_b = self.pc_rotation_scale(XYZid_b, 90)
                    self.summary_writer.add_mesh(tag='eval_pred_pc', vertices = XYZid_b, global_step=self.total_step_count*self.cfg.batchSize+batch)
                    
                    points = XYZid_b.squeeze(0).cpu().numpy()
                    # self.save_ply(points, os.path.join(input_batch["save_path"][batch], str(self.total_step_count*8+batch)+'_pre.ply'))
                    self.save_ply(points, os.path.join(input_batch["save_path"][batch], str(batch+1)+'_pre.ply'))
                    self.logger.info("Saving predict ply files...")
                    
                # sample pointcloud
                pc = input_batch["points"]
                for batch in range(pc.shape[0]):
                    pc_instance = pc[batch,:,:].view(1,3,-1)
                    self.summary_writer.add_mesh(tag='eval_sample_pc', vertices=pc_instance,global_step=self.total_step_count*self.cfg.batchSize+batch)
    def depth2point(self, input_depth):
        batch_size, _, _, _ = input_depth.size()
        XYZ = []
        X, Y = torch.meshgrid([
            torch.arange(self.cfg.outH), # [H,W]
            torch.arange(self.cfg.outW)]) # [H,W]
        X, Y = X.float(), Y.float()
        XY = torch.cat([
            X.repeat([self.outViewN, 1, 1]), 
            Y.repeat([self.outViewN, 1, 1])], dim=0) #[2V,H,W]
        XY = XY.unsqueeze(dim=0).to(self.cfg.device) # [1,2V,H,W] 
        XY = XY.repeat([batch_size,1,1,1])
        XYZ.append(XY)
        maskLogits = []
        for i in range(4):
            depth = input_depth[:,i,:,:].unsqueeze(1).permute(0,1,3,2)
            mask = depth != self.renderDepth
            XYZ.append(depth)
            maskLogits.append(mask)
        XYZ = torch.concat(XYZ, dim=1)
        maskLogits = torch.concat(maskLogits, dim=1)
        XYZid, ML = fuse3D(
                self.cfg, XYZ, maskLogits, self.cfg.fuseTrans_q, self.cfg.fuseTrans_t) # [B,3,VHW],[B,1,VHW]
        ML = ML > 0
        return XYZid, ML
    
    def pc_rotation_scale(self, pc, degrees, scale=1.0):
        theta = torch.deg2rad(torch.tensor(degrees, device=pc.device))

        # 构造旋转矩阵 [Batch, 3, 3]（广播到每个样本）
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.stack([
            torch.ones_like(cos_theta), torch.zeros_like(cos_theta), torch.zeros_like(cos_theta),
            torch.zeros_like(cos_theta), cos_theta, -sin_theta,
            torch.zeros_like(cos_theta), sin_theta, cos_theta
        ], dim=-1).reshape(-1, 3, 3)  # 兼容单角度或多角度
        
        # 构造缩放矩阵 [Batch, 3, 3]
        if isinstance(scale, (float, int)):
            scale = torch.full((rotation_matrix.size(0),), float(scale), device=pc.device)
        scale_matrix = torch.diag_embed(torch.stack([
            scale,  # X轴缩放（默认1.0）
            scale,                   # Y轴缩放
            scale                    # Z轴缩放
        ], dim=-1))  # 形状 [Batch, 3, 3]
        transform_matrix = torch.matmul(rotation_matrix, scale_matrix)
        # 应用旋转：矩阵乘法 (Batch, N, 3) @ (Batch, 3, 3) → (Batch, N, 3)
        return torch.matmul(pc, transform_matrix)
    def save_ply(self, points, filename):
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(filename, 'w') as f:
            f.write(header)
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
