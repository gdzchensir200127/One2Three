import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalRefinerLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
    # def forward(self, output, inputBatch):
    #     image_pred, mask_pred = output
    #     image_true = inputBatch['GT_signal_abs']
    #     mask_true = inputBatch['GT_signal_abs_mask']
    #     mask_binary = (mask_pred > 0.5).float()  # 二值化
    #     recon_loss = F.mse_loss(image_pred * mask_binary, image_true * mask_binary)
        
    #     # Mask 预测损失：二元交叉熵
    #     mask_loss = F.binary_cross_entropy(mask_pred, mask_true)
        
    #     # 总损失（权重可调）
    #     total_loss = recon_loss +  mask_loss  # 示例权重
    #     return total_loss, {
    #         "loss": total_loss,
    #         "recon_loss": recon_loss,
    #         "mask_loss": mask_loss
    #     }
    def forward(self, output, inputBatch):
        if self.options.signal_refiner == 'mse':
            # real_loss = F.mse_loss(output['I'], inputBatch['GT_signal_I'])
            # imag_loss = F.mse_loss(output['Q'], inputBatch['GT_signal_Q'])
            # magnitude_pre = torch.sqrt(output['I']**2 + output['Q']**2)
            # magnitude_GT = torch.sqrt(inputBatch['GT_signal_I']**2+ inputBatch['GT_signal_Q']**2)
            magnitude_loss = F.mse_loss(output, inputBatch['GT_signal_abs'])
            loss = magnitude_loss

            return loss, {
                "loss":loss,
                # "real_loss":real_loss,
                # "imag_loss":imag_loss,
                "magnitude_loss":magnitude_loss
            }
        elif self.options.signal_refiner == "weighted_mse":
            GT_signal_abs = inputBatch['GT_signal_abs']
            return self.weighted_mse_loss(output, GT_signal_abs)
    
    def weighted_mse_loss(self, output, GT_signal_abs, epsilon=1e-10):
        """
        加权均方误差损失函数，同时考虑样本和通道的权重。

        参数:
            output (Tensor): 模型输出的信号幅值，形状为 (batch_size, 2800, 9)。
            GT_signal_abs (Tensor): GT信号的幅值，形状为 (batch_size, 2800, 9)。
            epsilon (float): 避免除零的小常数。

        返回:
            loss (Tensor): 加权损失值。
        """

        # 计算每个通道的权重（与通道的整体幅值成反比）
        channel_norm = torch.sqrt(torch.sum(GT_signal_abs, dim=(0, 1)))  # 形状: (9,)
        channel_weights = 1 / (channel_norm + epsilon)  # 形状: (9,)
        channel_weights = channel_weights / channel_weights.sum()  # 归一化

        # 计算实部和虚部的 MSE 损失
        mse_loss = nn.MSELoss(reduction='none')  # 不进行 reduction
        # loss_real = mse_loss(output_real, target_real).mean(dim=1)  # 形状: (batch_size, 9)
        # loss_imag = mse_loss(output_imag, target_imag).mean(dim=1)  # 形状: (batch_size, 9)
        magnitude_loss = mse_loss(output, GT_signal_abs)
        # 加权损失
        weighted_loss = (
            channel_weights * magnitude_loss  # 形状: (batch_size, 9)
        ).sum()  # 标量
        loss = weighted_loss
        return weighted_loss,{
            'weighted_loss':weighted_loss,
            'loss':loss
        }