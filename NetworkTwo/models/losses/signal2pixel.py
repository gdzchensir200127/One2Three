import torch
import torch.nn as nn
import torch.nn.functional as F

class Signal2PixelLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.outH = 128
        self.outW = 128
        self.outViewN = 1
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, output, inputBatch):
        out = output['out']
        view_index = output['view_index']
        XYZ, maskLogit = out[0], out[1]
        XY = XYZ[:, :self.outViewN * 2, :, :]
        depth = XYZ[:, self.outViewN * 2:self.outViewN * 3, :,  :]
        mask = (maskLogit > 0)

        depthGT = inputBatch["project"][:,view_index,:,:].unsqueeze(1)
        maskGT = depthGT != 0
        # depthGT[~maskGT] = self.renderDepth
        XGT, YGT = torch.meshgrid([
            torch.arange(self.outH), # [H,W]
            torch.arange(self.outW)]) # [H,W]
        XGT, YGT = XGT.float(), YGT.float()
        XYGT = torch.cat([
            XGT.repeat([self.outViewN, 1, 1]), 
            YGT.repeat([self.outViewN, 1, 1])], dim=0) #[2V,H,W]
        XYGT = XYGT.unsqueeze(dim=0).to(XYZ.device) # [1,2V,H,W] 

        loss_XYZ = self.l1(XY, XYGT) + self.l1(depth.masked_select(mask),
                            depthGT.masked_select(mask))
        loss_mask = self.bce(maskLogit, maskGT.float())
        loss = loss_mask*self.options.weights.signal2pixel_mask + loss_XYZ*self.options.weights.signal2pixel_xyz

        return loss, {
            "loss":loss,
            "loss_XYZ":loss_XYZ,
            "loss_mask":loss_mask,
        }