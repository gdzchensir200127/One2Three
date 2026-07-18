import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict
import numpy as np
import scipy.io
from torchvision import transforms
from tqdm import tqdm
import traceback
import random
import os
from Fusion8 import MultimodalFusion 
from Decoder import FeatureEnhancementModule
from DepthEncoder import DepthSpatialEncoder, DepthTemporalEncoder
from SignalEncoder import SignalSpatialEncoder, SignalTemporalEncoder

#1,411,413
class Fnet(nn.Module):

    def __init__(self):
        super().__init__()

        #  Spatial and Temporal Encoder
        self.signal_spatial_encoder = SignalSpatialEncoder(in_channels=1, out_channels=64, hidden_dim=32)
        self.signal_temporal_encoder = SignalTemporalEncoder(in_channels=1, out_channels=64, hidden_dim=32)

        self.depth_spatial_encoder = DepthSpatialEncoder()
        self.depth_temporal_encoder = DepthTemporalEncoder(in_channels=1, out_channels=64, hidden_dim=32)

        self.multimodal_fusion = MultimodalFusion()
        
        self.signal_temporal_feature_linear=nn.Sequential(
            nn.Linear(1440, 720),
            nn.PReLU()
        )

        self.signal_spatial_feature_linear=nn.Sequential(
            nn.Linear(192, 96),
            nn.PReLU()
        )

        self.Decoder_I = FeatureEnhancementModule()
        self.Decoder_Q = FeatureEnhancementModule()
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

        self.upsample_I = nn.Upsample(
            scale_factor=(4, 1),  # (对H的缩放, 对W的缩放)
            mode='bilinear',      # 2D空间上采样需用bilinear模式（linear用于1D）
            align_corners=True    # 保持角落像素对齐（可选，根据需求调整）
        )
        self.upsample_Q = nn.Upsample(
            scale_factor=(4, 1),  # (对H的缩放, 对W的缩放)
            mode='bilinear',      # 2D空间上采样需用bilinear模式（linear用于1D）
            align_corners=True    # 保持角落像素对齐（可选，根据需求调整）
        )

        self.fc_out_I = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
        )
        self.fc_out_Q = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
        )

    def forward(self, signals, depths):
        # [B, 2880, 2, 96]
        b,l,c,d = signals.shape

        real_input=signals[:, :, 0, :]# [B,2880, 96]
        imag_input=signals[:, :, 1, :]# [B,2880, 96]

        real_input=self.avg_pool(real_input.permute(0,2,1)).permute(0,2,1) # [B, 720, 96]
        imag_input=self.avg_pool(imag_input.permute(0,2,1)).permute(0,2,1) # [B, 720, 96]

        signal_spatial_feature = self.signal_spatial_encoder(signals.permute(0,2,1,3))# [B, 2, 720, 96]

        signal_temporal_feature = self.signal_temporal_encoder(signals.permute(0,2,1,3))# [B, 2, 720, 96]

        depth_spatial_feature = self.depth_spatial_encoder(depths)  # [B, 720,96]

        depth_temporal_feature = self.depth_temporal_encoder(depth_spatial_feature)  # [B, 720, 96]

        signal_spatial_concat_feature = torch.cat([signal_spatial_feature[:, 0, :, :], signal_spatial_feature[:, 1, :, :]], dim=2)# [B, 720, 96*2]

        signal_temporal_concat_feature = torch.cat([signal_temporal_feature[:, 0, :, :], signal_temporal_feature[:, 1, :, :]], dim=1)# [B, 720*2, 96]

        # signal_temporal_concat_feature,signal_spatial_concat_feature,depth_temporal_feature,depth_spatial_feature
        # [B, 720*2, 96] ,[B, 720, 96*2],[B, 720, 96], [B, 720, 96] -->[B, 720, 96]

        signal_spatial_concat_feature=self.signal_spatial_feature_linear(signal_spatial_concat_feature) # [B, 720, 96]
        signal_temporal_concat_feature=self.signal_temporal_feature_linear(signal_temporal_concat_feature.permute(0,2,1)).permute(0,2,1) # [B, 720, 96]
        # [B, 720, 96] ,[B, 720, 96],[B, 720, 96], [B, 720, 96] -->[B, 720, 96]
        fused_feature = self.multimodal_fusion(signal_temporal_concat_feature,signal_spatial_concat_feature,depth_temporal_feature,depth_spatial_feature)  # [B, 720, 96]

        fused_feature=fused_feature.unsqueeze(dim=1)# [B, 1, 720, 96]
        real_input=real_input.unsqueeze(dim=1) # [B, 1, 720, 96]
        imag_input=imag_input.unsqueeze(dim=1) # [B, 1, 720, 96]
        real_result=self.Decoder_I(real_input,fused_feature)  # [B,1, 720, 96]
        imag_result=self.Decoder_Q(imag_input,fused_feature)  # [B,1, 720, 96]
        real_result=self.upsample_I(real_result)# [B,1, 2880, 96]
        imag_result=self.upsample_Q(imag_result)# [B,1, 2880, 96]

        output_I = self.fc_out_I(real_result)
        output_Q = self.fc_out_Q(imag_result)
        output_I = output_I.squeeze(1)  # [B, 2880, 96]
        output_Q = output_Q.squeeze(1)  # [B, 2880, 96]

        return (output_I,output_Q)

def validate_fnet():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测GPU是否可用，优先使用GPU 3，否则用CPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")  # 打印确认设备
    
    # 创建网络实例，并移动到指定设备（GPU 3或CPU）
    model = Fnet().to(device)
    print("成功创建Fnet网络实例并移动到目标设备")
    
    # 生成测试输入数据，并移动到与模型相同的设备
    batch_size = 1
    signals = torch.randn(batch_size, 2880, 2, 96).to(device)  # 移动到device
    depths = torch.randn(batch_size, 720, 576, 640).to(device)  # 移动到device
    
    print(f"测试输入形状:")
    print(f"  signals: {signals.shape} (设备: {signals.device})")  # 确认张量设备
    print(f"  depths: {depths.shape} (设备: {depths.device})")
    
    # 执行前向传播（此时模型和输入都在GPU 3，计算会在GPU上进行）
    try:
        with torch.no_grad():
            out_I, out_Q = model(signals, depths)
        
        print("\n前向传播成功!")
        print(f"输出形状:")
        print(f"  out_I: {out_I.shape} (设备: {out_I.device})")
        print(f"  out_Q: {out_Q.shape} (设备: {out_Q.device})")
        
        # 检查输出形状是否符合预期
        expected_shape = (batch_size, 2880, 96)
        assert out_I.shape == expected_shape, f"out_I形状不符合预期,得到{out_I.shape}，预期{expected_shape}"
        assert out_Q.shape == expected_shape, f"out_Q形状不符合预期,得到{out_Q.shape}，预期{expected_shape}"
        
        print("\n所有验证检查通过!")
        return True
        
    except Exception as e:
        print(f"\n前向传播过程中出现错误: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行验证
    validate_fnet()
    
    # 检查模型参数数量
    model = Fnet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数数量: {trainable_params:,}")
