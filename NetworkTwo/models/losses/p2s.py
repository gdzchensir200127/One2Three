import torch
import torch.nn as nn
import torch.nn.functional as F

class P2SLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')


    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2SModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        gt_I = targets["signal_real"][:,:,::self.options.downsample_factor] # 下采样到 8 * 9 * 280
        gt_Q = targets["signal_imag"][:,:,::self.options.downsample_factor]

        gt_I = F.normalize(gt_I,dim=2)
        gt_Q = F.normalize(gt_Q,dim=2)

        pred_I = outputs["I"]
        pred_Q = outputs["Q"]

        pred_I = F.normalize(pred_I,dim=2)
        pred_Q = F.normalize(pred_Q,dim=2)

        I_loss = self.l2_loss(gt_I, pred_I)
        Q_loss = self.l2_loss(gt_Q, pred_Q)

        loss = I_loss*0.5 + Q_loss*0.5

        return loss, {
            "loss": loss,
            "I_loss": I_loss,
            "Q_loss": Q_loss,
        }
