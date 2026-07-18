import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEnhancementModule(nn.Module):
    """
    输入:
        - input_tensor: 原始输入张量，形状为[B, 1, 720, 96]
        - feature_tensor: 从其他模块提取的特征张量，形状为[B, 1, 720, 96]
    输出:
        - enhanced_tensor: 增强后的输入张量，形状为[B, 1, 720, 96]
    """
    def __init__(self, in_channels=1, hidden_channels=8):
        super().__init__()
        
        # 从特征张量中提取关键信息
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        # 生成用于增强输入的注意力掩码
        self.attention_generator = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 使用sigmoid确保输出在0-1之间，作为注意力权重
        )
        
        # 残差连接的1x1卷积，用于调整通道数
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 最终的特征调整层
        self.final_adjust = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, input_tensor, feature_tensor):
        # 从特征张量中提取有用信息
        extracted_features = self.feature_extractor(feature_tensor)
        
        # 生成注意力掩码
        attention_mask = self.attention_generator(extracted_features)
        
        # 增强部分 = 输入 * (1 + 注意力掩码)
        enhanced_part = input_tensor * (1 + attention_mask)
        
        # 保留原始输入的信息
        residual = self.residual_conv(input_tensor)
        
        # 结合增强部分和残差连接
        combined = enhanced_part + residual
        
        # 最终调整并返回增强后的张量
        output = self.final_adjust(combined)
        
        return output

