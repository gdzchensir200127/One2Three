import torch
import numpy as np
from scipy.signal import gausspulse

XGT, YGT = torch.meshgrid([
    torch.arange(128), # [H,W]
    torch.arange(128)]) # [H,W]
XGT, YGT = XGT.float(), YGT.float()
XYGT = torch.cat([
    XGT.repeat([8, 1, 1]), 
    YGT.repeat([8, 1, 1])], dim=0) #[2V,H,W]
XYGT = XYGT.unsqueeze(dim=0) # [1,2V,H,W] 
pass
