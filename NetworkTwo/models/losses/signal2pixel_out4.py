import torch
import torch.nn as nn
import torch.nn.functional as F

class Signal2PixelOut4Loss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.outH = 128
        self.outW = 128
        self.outViewN = 4
        self.renderDepth = 0.781
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, output, inputBatch):
        depth, maskLogit = output[0], output[1]
        mask = (maskLogit > 0)

        depthGT = inputBatch["project"]
        maskGT = depthGT != 0
        depthGT[~maskGT] = self.renderDepth

        loss_XYZ = self.l1(depth.masked_select(mask),
                            depthGT.masked_select(mask))
        loss_mask = self.bce(maskLogit, maskGT.float())
        loss = loss_mask*self.options.weights.signal2pixel_mask + loss_XYZ*self.options.weights.signal2pixel_xyz

        return loss, {
            "loss":loss,
            "loss_XYZ":loss_XYZ,
            "loss_mask":loss_mask,
        }