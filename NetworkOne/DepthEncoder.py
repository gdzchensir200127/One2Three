import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict
import numpy as np
import scipy.io
from torchvision import transforms
import traceback

# [B, 720, 576, 640] -> [B,720,96]
class DepthSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 使用CNN提取空间特征
        self.spatial_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=4, padding=1),  # [16, 144, 160]
            nn.BatchNorm2d(16),
            nn.PReLU(),
            
            nn.Conv2d(16, 256, kernel_size=5, stride=4, padding=1),  # [256, 36, 40]
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=1)
        self.channel_compress = nn.Sequential(
        nn.Conv2d(256, 1, kernel_size=1),  # 256通道→1通道
        nn.BatchNorm2d(1),  # 对单通道做批归一化
        nn.PReLU()  # 激活函数
        )
        ##
        self.conv1=nn.Sequential(
            nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=4,  
            stride=3,
            padding=1
        ),
            nn.BatchNorm1d(1),
            nn.PReLU()
        )

        self.conv2=nn.Sequential(
            nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=7,  
            stride=5,
            padding=1
        ),
            nn.BatchNorm1d(1),
            nn.PReLU()
        )

    def forward(self, x):
        # x形状:[B, 720, 576, 640]
        
        # 保存原始形状信息
        B, T, H, W = x.shape  # T=720, H=576, W=640
        
        # 为灰度图添加通道维度，并重组维度以便处理所有时间步
        # 转换为 [B*T, 1, H, W] 形状，以便CNN处理
        x = x.reshape(B * T, 1, H, W)
        

        spatial_features = self.spatial_feature_extractor(x)  # [B*T, 256, 36, 40]
        
        # ##1.用mean压缩通道
        # spatial_features = spatial_features.reshape(spatial_features.shape[0], 256, -1)  # [B*T, 256, 36*40]
        # spatial_features = torch.mean(spatial_features, dim=1)# [B*T, 36*40]
        # spatial_features=spatial_features.reshape(B,T,1440)# [B,T, 36*40]
        # ##

        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=1)
        # [B*T, 256, 36, 40] → [B*T, 1, 36, 40]
        spatial_features = self.channel_compress(spatial_features)
        # [B*T, 1, 36*40] = [B*T, 1, 1440]
        spatial_features = spatial_features.flatten(2)  
        spatial_features = spatial_features.squeeze(1)  
        ##
        spatial_features=spatial_features.reshape(B * T, 1, 1440)
        spatial_features=self.conv1(spatial_features) # [B * T, 1, 480]
        spatial_features=self.conv2(spatial_features) # [B * T, 1, 96]
        spatial_features=spatial_features.reshape(B, T, 96) # [B, T, 96]
        return spatial_features

# [B, 720,96] -> [B, 720, 96]
class DepthTemporalEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, hidden_dim=32):
        super().__init__()
    
        self.conv1=nn.Sequential(
            nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=5,  
            stride=1,
            padding=2  
        ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        
        self.conv2=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=7,  
            stride=1,
            padding=3
        ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.conv3=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        ),
            nn.BatchNorm1d(out_channels)
        )

        self.residual_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )

        self.prelu = nn.PReLU()

        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=2)
        self.channel_compress = nn.Sequential(
        nn.Conv2d(out_channels, 1, kernel_size=1),  
        nn.BatchNorm2d(1),  # 对单通道做批归一化
        nn.PReLU()  # 激活函数
        )
        ##

    def forward(self, x):
        # 输入形状:[B, 720,96]
        B, T, S = x.shape  # B:批次大小, T:时间长度, S:空间位置数
    
        out_2= x.permute(0, 2, 1)  # 形状变为 [B, 96, 720]
        out_2 = out_2.reshape(B * S, 1, 720)  # 形状变为 [B*S, 1, 720]
        # 残差连接
        residual_out_2 = self.residual_conv(out_2)  # [B*S, out_channels, 720]

        # 第一个卷积块
        out_2 = self.conv1(out_2)  # [B*S, hidden_dim, 720]
        # 第二个卷积块
        out_2 = self.conv2(out_2)  # [B*S, hidden_dim, 720]
        # 第三个卷积块
        out_2 = self.conv3(out_2)  # [B*S, out_channels, 720]
        # 加上残差连接
        out_2 = out_2 + residual_out_2
        out_2 = self.prelu(out_2)  
        # 恢复原始维度 [B, 96, out_channels, 720]
        out_2 = out_2.reshape(B, S, -1, 720)
        # ## 1.用mean压缩通道
        # out_2 = torch.mean(out_2, dim=2)
        # ##
        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=2)
        out_2 = out_2.permute(0, 2, 1,3)  # [B, out_channels, 96, 720]
        out_2 = self.channel_compress(out_2).squeeze(1)  # [B, 96, 720]
        ##
        return out_2.permute(0, 2, 1)  # [B, 720, 96]
