import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from models.pointnet_utils import PointNetEncoder
class Discriminator(nn.Module):
  def __init__(self, options = None):
    super().__init__()
    self.pointEncoder = PointNetEncoder(global_feat=True,feature_transform=True,channel=3)
    self.actvn = F.relu
    self.fc_0 = nn.Linear(1024, 512)
    self.fc_1 = nn.Linear(512, 256)
    self.fc_2 = nn.Linear(256,128)
    self.fc_3 = nn.Linear(128, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self,x:torch.Tensor):
    '''
    x is cloud point B*1024*3
    '''
    x = x.permute(0,2,1)          # x [B*3*1024] pc
    x,_,_ = self.pointEncoder(x)  # x [B*1024] feature
    x = self.fc_0(x)
    x = self.fc_1(self.actvn(x))
    x = self.fc_2(self.actvn(x))
    x = self.fc_3(self.actvn(x)) # x [B*1]
    x = self.sigmoid(x)
    return x