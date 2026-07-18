import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class SignalRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder结构
        self.encoder = nn.Sequential(
            ConvBlock(9, 64, kernel_size=15, stride=2, padding=7),
            ConvBlock(64, 128, kernel_size=11, stride=2, padding=5),
            ConvBlock(128, 256, kernel_size=7, stride=2, padding=3),
            ConvBlock(256, 512, kernel_size=5, stride=1, padding=2),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1)
        )
        
        # Decoder结构
        self.decoder = nn.Sequential(
            DeconvBlock(1024, 512, kernel_size=3, stride=1, padding=1),
            DeconvBlock(512, 256, kernel_size=5, stride=1, padding=2),
            DeconvBlock(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            DeconvBlock(128, 64, kernel_size=11, stride=2, padding=5, output_padding=1),
            DeconvBlock(64, 9, kernel_size=15, stride=2, padding=7, output_padding=1, final_layer=True)
        )

    def forward(self, x):
        # 输入维度转换 [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)  # 2800*9 -> 9*2800
        
        # 编码-解码处理
        x = self.encoder(x)
        x = self.decoder(x)
        
        # 输出维度恢复 [B, C, L] -> [B, L, C]
        return x.permute(0, 2, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, 
                 output_padding=0, final_layer=False):
        super().__init__()
        
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, 
            stride=stride, 
            padding=padding,
            output_padding=output_padding
        )
        
        self.activation = None if final_layer else nn.LeakyReLU(0.2, inplace=True)
        self.bn = None if final_layer else nn.BatchNorm1d(out_ch)
        
    def forward(self, x):
        x = self.deconv(x)
        if self.bn!=None:
             x = self.bn(x)
        if self.activation!=None:
            x=self.activation(x)
        return x
        
# class ChannelWeightedLoss(nn.Module):
#     def __init__(self, alpha=0.7, eps=1e-6):
#         super().__init__()
#         self.alpha = alpha
#         self.eps = eps
        
#     def forward(self, pred, target):
#         # 通道能量计算 (保持原始维度[B, L, C])
#         channel_energy = torch.mean(target**2, dim=1)  # [B, C]
#         weights = 1.0 / (channel_energy + self.eps)     # 能量倒数加权
#         weights = weights / torch.sum(weights, dim=1, keepdim=True)  # 归一化
        
#         # 加权MSE损失
#         weighted_mse = torch.mean(weights.unsqueeze(1) * (pred - target)**2)
        
#         # 结构相似性损失
#         ssim_loss = 1.0 - self._multi_ssim(pred, target)
        
#         return self.alpha*ssim_loss + (1-self.alpha)*weighted_mse
    
#     def _multi_ssim(self, x, y, window_size=11, sigma=1.5):
#         # 多通道SSIM实现
#         channels = x.shape[-1]
#         total_ssim = 0.0
        
#         for c in range(channels):
#             total_ssim += self._ssim(x[...,c], y[...,c], window_size, sigma)
            
#         return total_ssim / channels
    
#     def _ssim(self, x, y, window_size, sigma):
#         # 单通道SSIM计算
#         # ... (具体实现代码)
#         return ssim_value

# # 验证维度正确性
# def test_dimensions():
#     model = EnergyAwareED()
#     dummy_input = torch.randn(8, 2800, 9)  # [batch, length, channels]
#     output = model(dummy_input)
    
#     print(f"输入尺寸: {dummy_input.shape}")
#     print(f"输出尺寸: {output.shape}")
#     assert output.shape == dummy_input.shape, "维度不匹配！"
    
# if __name__ == "__main__":
#     test_dimensions()  # 应输出匹配的维度
# class SignalRefiner(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # ----------- 增强的编码器 -----------
#         self.encoder = nn.Sequential(
#             # 初始卷积块（通道数翻倍）
#             nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1)),  # [B, 128, 2800, 9]
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),  # 增加额外卷积层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),          # [B, 128, 1400, 9]
            
#             # 残差块1（通道数翻倍）
#             ResidualBlock(128, 256),                                 # [B, 256, 1400, 9]
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),          # [B, 256, 700, 9]
            
#             # 残差块2（通道数翻倍）
#             ResidualBlock(256, 512),                                 # [B, 512, 700, 9]
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),          # [B, 512, 350, 9]
            
#             # 残差块3（保持通道数）
#             ResidualBlock(512, 512),                                 # [B, 512, 350, 9]
#         )
        
#         # ----------- 增强的解码器分支1：图像重建 -----------
#         self.decoder_image = nn.Sequential(
#             # 上采样模块1（通道数减半）
#             nn.ConvTranspose2d(512, 256, kernel_size=(2, 1), stride=(2, 1)),  # [B, 256, 700, 9]
#             ResidualBlock(256, 256),                                          # [B, 256, 700, 9]
            
#             # 上采样模块2（通道数减半）
#             nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1)),  # [B, 128, 1400, 9]
#             ResidualBlock(128, 128),                                          # [B, 128, 1400, 9]
            
#             # 上采样模块3（通道数减半）
#             nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1)),   # [B, 64, 2800, 9]
#             ResidualBlock(64, 64),                                            # [B, 64, 2800, 9]
            
#             # 最终输出层
#             nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1)),             # [B, 1, 2800, 9]
#         )
        
#         # ----------- 增强的解码器分支2：Mask 预测 -----------
#         self.decoder_mask = nn.Sequential(
#             # 上采样模块1（通道数减半）
#             nn.ConvTranspose2d(512, 256, kernel_size=(2, 1), stride=(2, 1)),  # [B, 256, 700, 9]
#             ResidualBlock(256, 256),                                          # [B, 256, 700, 9]
            
#             # 上采样模块2（通道数减半）
#             nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1)),  # [B, 128, 1400, 9]
#             ResidualBlock(128, 128),                                          # [B, 128, 1400, 9]
            
#             # 上采样模块3（通道数减半）
#             nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1)),   # [B, 64, 2800, 9]
#             ResidualBlock(64, 64),                                            # [B, 64, 2800, 9]
            
#             # 最终输出层
#             nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1)),             # [B, 1, 2800, 9]
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1)                    # [B, 1, 2800, 9]
#         x_encoded = self.encoder(x)           # [B, 512, 350, 9]
#         image_recon = self.decoder_image(x_encoded).squeeze(1)  # [B, 2800, 9]
#         mask_pred = self.decoder_mask(x_encoded).squeeze(1)     # [B, 2800, 9]
#         return image_recon, mask_pred

# # ----------- 残差块定义 -----------
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
#         )
#         self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()
        
#     def forward(self, x):
#         return F.relu(self.conv(x) + self.skip(x))
# # class SignalRefiner(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # ----------- 共享编码器 -----------
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),  # [B, 64, 2800, 9]
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # [B, 64, 1400, 9]
#             nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),# [B, 128, 1400, 9]
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # [B, 128, 700, 9]
#             nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),# [B, 256, 700, 9]
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # [B, 256, 350, 9]
#         )
        
#         # ----------- 解码器分支1：图像重建 -----------
#         self.decoder_image = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1)),  # [B, 128, 700, 9]
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1)),   # [B, 64, 1400, 9]
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1)),    # [B, 32, 2800, 9]
#             nn.ReLU(),
#             nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1)),             # [B, 1, 2800, 9]
#         )
        
#         # ----------- 解码器分支2：Mask 预测 -----------
#         self.decoder_mask = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1)),  # [B, 128, 700, 9]
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1)),   # [B, 64, 1400, 9]
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1)),    # [B, 32, 2800, 9]
#             nn.ReLU(),
#             nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1)),             # [B, 1, 2800, 9]
#             nn.Sigmoid()  # 输出 Mask 的概率图 [0, 1]
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x_encoded = self.encoder(x)
#         image_recon = self.decoder_image(x_encoded)  # 重建图像
#         mask_pred = self.decoder_mask(x_encoded)     # 预测 Mask
#         return image_recon.squeeze(1), mask_pred.squeeze(1)

# class SignalRefiner(nn.Module):
#     def __init__(self, input_channels=9, output_channels=9, hidden_channels=64, kernel_size=5):
#         super(SignalRefiner, self).__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size, padding=kernel_size//2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size, padding=kernel_size//2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose1d(hidden_channels, output_channels, kernel_size=2, stride=2)
#             # 移除 Sigmoid 激活函数
#         )
    
#     def forward(self, x):
#         # Encoder
#         x = x.permute(0, 2, 1)  # (batch_size, channels, seq_len)
#         x = self.encoder(x)
        
#         # Decoder
#         x = self.decoder(x)
#         x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
#         return x


# # 定义复数信号的MSE损失函数
# def complex_mse_loss(output, target):
#     real_loss = F.mse_loss(output.real, target.real)
#     imag_loss = F.mse_loss(output.imag, target.imag)
#     return real_loss + imag_loss
# # 定义网络结构
# class SignalRefiner(nn.Module):
#     def __init__(self):
#         super(SignalRefiner, self).__init__()
        
#         # # 定义卷积层，输入通道数为18（9个复数的实部和虚部），输出通道数为18
#         # self.conv1 = nn.Conv1d(in_channels=18, out_channels=18, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv1d(in_channels=18, out_channels=18, kernel_size=3, padding=1)
        
#         # # 最后的卷积层，输出通道数为18，保持与输入相同的维度
#         # self.conv3 = nn.Conv1d(in_channels=18, out_channels=18, kernel_size=3, padding=1)
#         # 编码部分（下采样）
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels=18, out_channels=36, kernel_size=3, padding=1),  # 输入通道 18，输出通道 36
#             nn.ReLU(),
#             nn.Conv1d(in_channels=36, out_channels=72, kernel_size=3, padding=1),  # 输入通道 36，输出通道 72
#             nn.ReLU(),
#         )
        
#         # 解码部分（上采样）
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=72, out_channels=36, kernel_size=3, padding=1),  # 输入通道 72，输出通道 36
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=36, out_channels=18, kernel_size=3, padding=1),  # 输入通道 36，输出通道 18
#             nn.ReLU(),
#         )
        
#         # 最后的卷积层，输出通道数为18，保持与输入相同的维度
#         self.final_conv = nn.Conv1d(in_channels=18, out_channels=18, kernel_size=3, padding=1)
        
#     def forward(self, I, Q):
#         x = self.complex_to_channels(I, Q)
#         # 输入x的形状为 (batch_size, 18, 2800)
#         # x = F.relu(self.conv1(x))
#         # x = F.relu(self.conv2(x))
#         # x = self.conv3(x)
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = self.final_conv(x)
#         return self.channels_to_complex(x)

#     # 将复数信号拆分为实部和虚部
#     def complex_to_channels(self, I, Q):
#         # 将实部和虚部拼接在一起，形状为 (batch_size, 2800, 18)
#         combined = torch.cat([I, Q], dim=-1)
        
#         # 将通道维度移到第1维，形状变为 (batch_size, 18, 2800)
#         combined = combined.permute(0, 2, 1)
        
#         return combined

#     # 将网络输出转换回复数信号
#     def channels_to_complex(self, output):
#         # output的形状为 (batch_size, 18, 2800)
#         # 将通道维度移回最后一维，形状变为 (batch_size, 2800, 18)
#         output = output.permute(0, 2, 1)
        
#         # 将前9个通道作为实部，后9个通道作为虚部
#         real_part = output[:, :, :9]
#         imag_part = output[:, :, 9:]
        
#         # 将实部和虚部组合成复数信号，形状为 (batch_size, 2800, 9)
#         return {
#             'I': real_part,
#             'Q': imag_part
#         }

# # 定义复数信号的MSE损失函数
# def complex_mse_loss(output, target):
#     real_loss = F.mse_loss(output.real, target.real)
#     imag_loss = F.mse_loss(output.imag, target.imag)
#     return real_loss + imag_loss

# 示例使用
if __name__ == "__main__":
    # 假设输入是一个batch_size为1的复数信号
    batch_size = 1
    input_signal = torch.randn(batch_size,  2800, 9)
    
    
    # 初始化网络
    model = SignalRefiner().cuda()
    
    # 前向传播
    # output_signal = model(input_signal)
    
    # 打印输出形状
    # print("Output signal shape:", output_signal.shape)  # 输出形状应为 (batch_size, 2800, 9)
    summary(model, input_size=(2800,9))
    # # 假设目标信号与输入信号相同（仅用于示例）
    # target_signal = input_signal
    
    # # 计算损失
    # loss = complex_mse_loss(output_signal, target_signal)
    # print("Loss:", loss.item())