import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict
import numpy as np
import scipy.io
from torchvision import transforms
import traceback

# [B, 2, 2880, 96] -> [B, 2, 720, 96]
class SignalSpatialEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, hidden_dim=32):
        super().__init__()
        
        self.conv1_I=nn.Sequential(
            nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,  
            stride=1,
            padding=1  
        ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        
        self.conv2_I=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=5,  
            stride=1,
            padding=2
        ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.conv3_I=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
            nn.BatchNorm1d(out_channels)
        )

        self.conv1_Q=nn.Sequential(
            nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,  
            stride=1,
            padding=1  
        ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        
        self.conv2_Q=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=5,  
            stride=1,
            padding=2
        ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.conv3_Q=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
            nn.BatchNorm1d(out_channels)
        )

        self.residual_conv_I = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.residual_conv_Q = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.prelu_I = nn.PReLU()
        self.prelu_Q = nn.PReLU()

        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=2)
        self.channel_compress_I = nn.Sequential(
        nn.Conv2d(out_channels, 1, kernel_size=1),  
        nn.BatchNorm2d(1),  # 对单通道做批归一化
        nn.PReLU()  # 激活函数
        )

        self.channel_compress_Q = nn.Sequential(
        nn.Conv2d(out_channels, 1, kernel_size=1),  
        nn.BatchNorm2d(1),  # 对单通道做批归一化
        nn.PReLU()  # 激活函数
        )
        ##
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        # 输入形状:[B, 2, 2880, 96]
        
        # 调整维度以适应Conv1d的输入要求 [B, C, W]
        # 这里我们将时间步视为批次维度的一部分，对每个时间步的空间特征进行处理
        B, C, T, S = x.shape  # B:批次, C:实部/虚部,T:时间, S:空间

        out_1 = torch.chunk(x, 2, dim=1)  # 分实部和虚部
        out_2_I = out_1[0].squeeze(1) #[B, 2880, 96]
        out_2_Q = out_1[1].squeeze(1)

        out_2_I = out_2_I.reshape(B * T, 1, S)  # 形状变为 [B*T, 1, 96]
        out_2_Q = out_2_Q.reshape(B * T, 1, S)  # 形状变为 [B*T, 1, 96]
        
        # 残差连接
        residual_I = self.residual_conv_I(out_2_I)  # [B*T, out_channels, 96]
        residual_Q = self.residual_conv_Q(out_2_Q)  # [B*T, out_channels, 96]
        
        # 第一个卷积块
        out_2_I = self.conv1_I(out_2_I)  # [B*T, hidden_dim, 96]
        out_2_Q = self.conv1_Q(out_2_Q)  # [B*T, hidden_dim, 96]

        # 第二个卷积块
        out_2_I = self.conv2_I(out_2_I)  # [B*T, hidden_dim, 96]
        out_2_Q = self.conv2_Q(out_2_Q)  # [B*T, hidden_dim, 96]
        
        # 第三个卷积块
        out_2_I = self.conv3_I(out_2_I)  # [B*T, out_channels, 96]
        out_2_Q = self.conv3_Q(out_2_Q)  # [B*T, out_channels, 96]

        # 加上残差连接
        out_2_I = out_2_I + residual_I
        out_2_Q = out_2_Q + residual_Q
        out_2_I = self.prelu_I(out_2_I)  
        out_2_Q = self.prelu_Q(out_2_Q)  
        
        # 恢复原始维度 [B, 2880, out_channels, 96]
        out_2_I = out_2_I.reshape(B, T, -1, S)
        out_2_Q = out_2_Q.reshape(B, T, -1, S)
        
        # ## 1.用mean压缩通道
        # out_2_I = torch.mean(out_2_I, dim=2)
        # out_2_Q = torch.mean(out_2_Q , dim=2)
        # ##
        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=2)
        out_2_I = out_2_I.permute(0, 2, 1,3)  # [B, out_channels, 2880, 96]
        out_2_Q = out_2_Q.permute(0, 2, 1,3)  # [B, out_channels, 2880, 96]
        out_2_I = self.channel_compress_I(out_2_I).squeeze(1)  # [B, 2880, 96]
        out_2_Q = self.channel_compress_Q(out_2_Q).squeeze(1)  # [B, 2880, 96]
        ##
        # 在第1个维度上拼接
        combined_tensor = torch.cat([self.avg_pool(out_2_I.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1), self.avg_pool(out_2_Q.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1)], dim=1)
        # 输出形状:[B, 2, 720, 96]
        return combined_tensor

# class SignalSpatialEncoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=64, hidden_dim=32):
#         super().__init__()
        
#         # 共享卷积模块（I路和Q路共用）
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=hidden_dim,
#                 kernel_size=3,  
#                 stride=1,
#                 padding=1  
#             ),
#             nn.BatchNorm1d(hidden_dim),
#             nn.PReLU()
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_dim,
#                 out_channels=hidden_dim,
#                 kernel_size=5,  
#                 stride=1,
#                 padding=2
#             ),
#             nn.BatchNorm1d(hidden_dim),
#             nn.PReLU()
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_dim,
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.BatchNorm1d(out_channels)
#         )

#         # 共享残差连接卷积
#         self.residual_conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1
#         )
        
#         self.prelu = nn.PReLU()  # 共享激活函数

#         # 共享通道压缩模块
#         self.channel_compress = nn.Sequential(
#             nn.Conv2d(out_channels, 1, kernel_size=1),  
#             nn.BatchNorm2d(1),
#             nn.PReLU()
#         )

#         self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

#     def forward(self, x):
#         # 输入形状:[B, 2, 2880, 96]，其中第2维是I/Q通道
        
#         B, C, T, S = x.shape  # B:批次, C:实部/虚部(2), T:时间, S:空间

#         # 分离I路和Q路（保持通道分离但共享处理模块）
#         out_I = x[:, 0, :, :]  # [B, 2880, 96]
#         out_Q = x[:, 1, :, :]  # [B, 2880, 96]

#         # 调整维度适应Conv1d: [B*T, 1, S]
#         out_I = out_I.reshape(B * T, 1, S)
#         out_Q = out_Q.reshape(B * T, 1, S)
        
#         # 共享残差连接计算
#         residual_I = self.residual_conv(out_I)  # [B*T, out_channels, 96]
#         residual_Q = self.residual_conv(out_Q)  # [B*T, out_channels, 96]
        
#         # 共享卷积块处理
#         out_I = self.conv1(out_I)  # [B*T, hidden_dim, 96]
#         out_Q = self.conv1(out_Q)
        
#         out_I = self.conv2(out_I)  # [B*T, hidden_dim, 96]
#         out_Q = self.conv2(out_Q)
        
#         out_I = self.conv3(out_I)  # [B*T, out_channels, 96]
#         out_Q = self.conv3(out_Q)

#         # 共享残差连接和激活
#         out_I = out_I + residual_I
#         out_Q = out_Q + residual_Q
#         out_I = self.prelu(out_I)  
#         out_Q = self.prelu(out_Q)  
        
#         # 恢复原始维度 [B, T, out_channels, S]
#         out_I = out_I.reshape(B, T, -1, S)
#         out_Q = out_Q.reshape(B, T, -1, S)
        
#         # 共享通道压缩（替代原来的mean操作）
#         out_I = out_I.permute(0, 2, 1, 3)  # [B, out_channels, T, S]
#         out_Q = out_Q.permute(0, 2, 1, 3)
        
#         out_I = self.channel_compress(out_I).squeeze(1)  # [B, T, S]
#         out_Q = self.channel_compress(out_Q).squeeze(1)
        
#         # 池化并拼接
#         combined_tensor = torch.cat([
#             self.avg_pool(out_I.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1),
#             self.avg_pool(out_Q.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1)
#         ], dim=1)
        
#         # 输出形状:[B, 2, 720, 96]（与原输出一致）
#         return combined_tensor

# [B, 2, 2880, 96] -> [B, 2, 720, 96]
class SignalTemporalEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, hidden_dim=32):
        super().__init__()
        
        self.conv1_I=nn.Sequential(
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
        
        self.conv2_I=nn.Sequential(
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

        self.conv3_I=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        ),
            nn.BatchNorm1d(out_channels)
        )

        self.conv1_Q=nn.Sequential(
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
        
        self.conv2_Q=nn.Sequential(
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

        self.conv3_Q=nn.Sequential(
            nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        ),
            nn.BatchNorm1d(out_channels)
        )

        self.residual_conv_I = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.residual_conv_Q = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.prelu_I = nn.PReLU()
        self.prelu_Q = nn.PReLU()

        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=2)
        self.channel_compress_I = nn.Sequential(
        nn.Conv2d(out_channels, 1, kernel_size=1),  
        nn.BatchNorm2d(1),  # 对单通道做批归一化
        nn.PReLU()  # 激活函数
        )

        self.channel_compress_Q = nn.Sequential(
        nn.Conv2d(out_channels, 1, kernel_size=1),  
        nn.BatchNorm2d(1),  # 对单通道做批归一化
        nn.PReLU()  # 激活函数
        )
        ##
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        # 输入形状:[B, 2, 2880, 96]
        B, C, T, S = x.shape  # B:批次大小, T:时间长度, S:空间位置数
        
        out_1 = torch.chunk(x, 2, dim=1)  # 分实部和虚部
        out_2_I = out_1[0].squeeze(1) #[B, 2880, 96]
        out_2_Q = out_1[1].squeeze(1)

        out_2_I= out_2_I.permute(0, 2, 1)  # 形状变为 [B, 96, 2880]
        out_2_Q= out_2_Q.permute(0, 2, 1)  # 形状变为 [B, 96, 2880]

        out_2_I=self.avg_pool(out_2_I)  # [B, 96, 720]
        out_2_Q=self.avg_pool(out_2_Q)  # [B, 96, 720]  

        out_2_I = out_2_I.reshape(B * S, 1, 720)  # 形状变为 [B*S, 1, 720]
        out_2_Q = out_2_Q.reshape(B * S, 1, 720)  # 形状变为 [B*S, 1, 720]

        # 残差连接
        residual_I = self.residual_conv_I(out_2_I)  # [B*S, out_channels, 720]
        residual_Q = self.residual_conv_Q(out_2_Q)  # [B*S, out_channels, 720]
        
        # 第一个卷积块
        out_2_I = self.conv1_I(out_2_I)  # [B*S, hidden_dim, 720]
        out_2_Q = self.conv1_Q(out_2_Q)  # [B*S, hidden_dim, 720]

        # 第二个卷积块
        out_2_I = self.conv2_I(out_2_I)  # [B*S, hidden_dim, 720]
        out_2_Q = self.conv2_Q(out_2_Q)  # [B*S, hidden_dim, 720]
        
        # 第三个卷积块
        out_2_I = self.conv3_I(out_2_I)  # [B*S, out_channels, 720]
        out_2_Q = self.conv3_Q(out_2_Q)  # [B*S, out_channels, 720]

        # 加上残差连接
        out_2_I = out_2_I + residual_I
        out_2_Q = out_2_Q + residual_Q
        out_2_I = self.prelu_I(out_2_I)  
        out_2_Q = self.prelu_Q(out_2_Q)  
        
        # 恢复原始维度 [B, 96, out_channels, 720]
        out_2_I = out_2_I.reshape(B, S, -1, 720)
        out_2_Q = out_2_Q.reshape(B, S, -1, 720)

        # ## 1.用mean压缩通道
        # out_2_I = torch.mean(out_2_I, dim=2)
        # out_2_Q = torch.mean(out_2_Q, dim=2)
        # ##
        ##2.用1x1卷积压缩通道替代原来的torch.mean(dim=2)
        out_2_I = out_2_I.permute(0, 2, 1,3)  # [B, out_channels, 96, 720]
        out_2_Q = out_2_Q.permute(0, 2, 1,3)  # [B, out_channels, 96, 720]
        out_2_I = self.channel_compress_I(out_2_I).squeeze(1)  # [B, 96, 720]
        out_2_Q = self.channel_compress_Q(out_2_Q).squeeze(1)  # [B, 96, 720]
        ##
        # 在第1个维度上拼接
        combined_tensor = torch.cat([out_2_I.permute(0, 2, 1).unsqueeze(1), out_2_Q.permute(0, 2, 1).unsqueeze(1)], dim=1)

        return combined_tensor
# class SignalTemporalEncoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=64, hidden_dim=32):
#         super().__init__()
        
#         # 共享卷积模块 - 替代原来I路和Q路各自的卷积层
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=hidden_dim,
#                 kernel_size=5,  
#                 stride=1,
#                 padding=2  
#             ),
#             nn.BatchNorm1d(hidden_dim),
#             nn.PReLU()
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_dim,
#                 out_channels=hidden_dim,
#                 kernel_size=7,  
#                 stride=1,
#                 padding=3
#             ),
#             nn.BatchNorm1d(hidden_dim),
#             nn.PReLU()
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_dim,
#                 out_channels=out_channels,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2
#             ),
#             nn.BatchNorm1d(out_channels)
#         )

#         # 共享残差连接卷积
#         self.residual_conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1
#         )
        
#         # 共享激活函数
#         self.prelu = nn.PReLU()

#         # 共享通道压缩模块
#         self.channel_compress = nn.Sequential(
#             nn.Conv2d(out_channels, 1, kernel_size=1),  
#             nn.BatchNorm2d(1),
#             nn.PReLU()
#         )

#         self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

#     def forward(self, x):
#         # 输入形状:[B, 2, 2880, 96]
#         B, C, T, S = x.shape  # B:批次大小, T:时间长度, S:空间位置数
        
#         # 分实部(I)和虚部(Q)
#         out_1 = torch.chunk(x, 2, dim=1)  
#         out_2_I = out_1[0].squeeze(1)  # [B, 2880, 96]
#         out_2_Q = out_1[1].squeeze(1)  # [B, 2880, 96]

#         # 调整维度用于1D卷积
#         out_2_I = out_2_I.permute(0, 2, 1)  # [B, 96, 2880]
#         out_2_Q = out_2_Q.permute(0, 2, 1)  # [B, 96, 2880]

#         # 平均池化
#         out_2_I = self.avg_pool(out_2_I)  # [B, 96, 720]
#         out_2_Q = self.avg_pool(out_2_Q)  # [B, 96, 720]  

#         # 调整形状用于共享卷积
#         out_2_I = out_2_I.reshape(B * S, 1, 720)  # [B*S, 1, 720]
#         out_2_Q = out_2_Q.reshape(B * S, 1, 720)  # [B*S, 1, 720]

#         # 共享残差连接
#         residual_I = self.residual_conv(out_2_I)  # [B*S, out_channels, 720]
#         residual_Q = self.residual_conv(out_2_Q)  # [B*S, out_channels, 720]
        
#         # 共享卷积块处理
#         out_2_I = self.conv1(out_2_I)  # [B*S, hidden_dim, 720]
#         out_2_Q = self.conv1(out_2_Q)
        
#         out_2_I = self.conv2(out_2_I)  # [B*S, hidden_dim, 720]
#         out_2_Q = self.conv2(out_2_Q)
        
#         out_2_I = self.conv3(out_2_I)  # [B*S, out_channels, 720]
#         out_2_Q = self.conv3(out_2_Q)

#         # 加上残差连接并激活
#         out_2_I = self.prelu(out_2_I + residual_I)
#         out_2_Q = self.prelu(out_2_Q + residual_Q)
        
#         # 恢复原始维度
#         out_2_I = out_2_I.reshape(B, S, -1, 720)  # [B, 96, out_channels, 720]
#         out_2_Q = out_2_Q.reshape(B, S, -1, 720)  # [B, 96, out_channels, 720]

#         # 共享通道压缩
#         out_2_I = out_2_I.permute(0, 2, 1, 3)  # [B, out_channels, 96, 720]
#         out_2_Q = out_2_Q.permute(0, 2, 1, 3)  # [B, out_channels, 96, 720]
        
#         out_2_I = self.channel_compress(out_2_I).squeeze(1)  # [B, 96, 720]
#         out_2_Q = self.channel_compress(out_2_Q).squeeze(1)  # [B, 96, 720]

#         # 拼接I路和Q路结果
#         combined_tensor = torch.cat(
#             [out_2_I.permute(0, 2, 1).unsqueeze(1), 
#              out_2_Q.permute(0, 2, 1).unsqueeze(1)], 
#             dim=1
#         )

#         return combined_tensor