import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy

class ConvModule(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None,in_ch=9,out_ch=32,):
    self.options = options
    super(ConvModule,self).__init__()
    self.in_ch= in_ch
    self.out_ch= out_ch
    self.actvn = nn.ReLU()
    
    self.conv3 = nn.Conv2d(self.in_ch,self.out_ch,3,padding=1)
    self.conv5 = nn.Conv2d(self.in_ch,self.out_ch,5,padding=2)
    self.conv7 = nn.Conv2d(self.in_ch,self.out_ch,7,padding=3)
    self.conv9 = nn.Conv2d(self.in_ch,self.out_ch,9,padding=4)
    

  def forward(self,sig):
    out = self.actvn(self.conv3(sig)+self.conv5(sig)+self.conv7(sig)+self.conv9(sig))
    return out
  
class ComplexModule(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None,view=1):
    self.options = options
    super(ComplexModule,self).__init__()
    
    self.actvn = nn.ReLU()
    self.conv_real = nn.Conv2d(1,1,3,padding=1)
    self.conv_imag = nn.Conv2d(1,1,3,padding=1)

  def forward(self,real,imag):
    real = torch.unsqueeze(real,dim=1)
    imag = torch.unsqueeze(imag,dim=1)
    out_real = self.actvn(self.conv_real(real)-self.conv_imag(imag)).squeeze(dim=1)
    out_imag = self.actvn(self.conv_imag(imag)+self.conv_real(real)).squeeze(dim=1)
    return out_real, out_imag
  
class PoolModule(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None,poolFunc='Max',kernelsize_l=1,kernelsize_w=2):
    self.options = options
    super(PoolModule,self).__init__()
    
    if poolFunc=='Max':
      self.pool_real = nn.MaxPool2d((kernelsize_l,kernelsize_w))
      self.pool_imag = nn.MaxPool2d((kernelsize_l,kernelsize_w))
    else:
      self.pool_real = nn.AvgPool2d((kernelsize_l,kernelsize_w))
      self.pool_imag = nn.AvgPool2d((kernelsize_l,kernelsize_w))

  def forward(self,real,imag):
    out_real = self.pool_real(real)
    out_imag = self.pool_imag(imag)
    return out_real, out_imag
  
class ComplexEncoder(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None,view=4,):
    self.options = options
    super(ComplexEncoder,self).__init__()
    self.view= view
    self.actvn = nn.ReLU()
    
    # Conv module
    self.conv1Real = ConvModule(in_ch=9,out_ch=32)
    self.conv1Imag = ConvModule(in_ch=9,out_ch=32)
    
    self.conv2Real = ConvModule(in_ch=32,out_ch=64)
    self.conv2Imag = ConvModule(in_ch=32,out_ch=64)

    self.conv3Real = ConvModule(in_ch=64,out_ch=128)
    self.conv3Imag = ConvModule(in_ch=64,out_ch=128)
    
    self.conv4Real = ConvModule(in_ch=128,out_ch=256)
    self.conv4Imag = ConvModule(in_ch=128,out_ch=256)

    self.conv5Real = ConvModule(in_ch=256,out_ch=512)
    self.conv5Imag = ConvModule(in_ch=256,out_ch=512)

    # complex module
    self.comp1 = ComplexModule()
    self.comp2 = ComplexModule()
    self.comp3 = ComplexModule()
    self.comp4 = ComplexModule()
    self.comp5 = ComplexModule()

    # Pooling
    self.pool1 = PoolModule(poolFunc='Max',kernelsize_l=1,kernelsize_w=2)
    self.pool2 = PoolModule(poolFunc='Max',kernelsize_l=1,kernelsize_w=2)
    self.pool3 = PoolModule(poolFunc='Max',kernelsize_l=1,kernelsize_w=2)
    self.pool4 = PoolModule(poolFunc='Max',kernelsize_l=1,kernelsize_w=2)
    # self.pool5 = PoolModule(poolFunc='Max',kernelsize_l=1,kernelsize_w=2)
    

  def forward(self,real,imag):
    #Layer 1 : [Batch,Sample,Antenna]->[8,9,2800] -> [8,32,1400]
    out_1_real = self.conv1Real(real)
    out_1_imag = self.conv1Imag(imag)
    out_1_real,out_1_imag = self.comp1(out_1_real,out_1_imag)
    out_1_real,out_1_imag = self.pool1(out_1_real,out_1_imag)

    #Layer 2 : -> [8,View,64,700]
    out_2_real = self.conv2Real(out_1_real)
    out_2_imag = self.conv2Imag(out_1_imag)
    out_2_real,out_2_imag = self.comp2(out_2_real,out_2_imag)
    out_2_real,out_2_imag = self.pool2(out_2_real,out_2_imag)

    #Layer 3 : -> [8,128,350]
    out_3_real = self.conv3Real(out_2_real)
    out_3_imag = self.conv3Imag(out_2_imag)
    out_3_real,out_3_imag = self.comp3(out_3_real,out_3_imag)
    out_3_real,out_3_imag = self.pool3(out_3_real,out_3_imag)

    #Layer 4 : -> [8,256,175]
    out_4_real = self.conv4Real(out_3_real)
    out_4_imag = self.conv4Imag(out_3_imag)
    out_4_real,out_4_imag = self.comp4(out_4_real,out_4_imag)
    out_4_real,out_4_imag = self.pool4(out_4_real,out_4_imag)

    #Layer 5 : -> [8,512,175]
    out_5_real = self.conv5Real(out_4_real)
    out_5_imag = self.conv5Imag(out_4_imag)
    out_5_real,out_5_imag = self.comp5(out_5_real,out_5_imag)

    out=[]
    out.append(out_5_real)
    out.append(out_5_imag)
    out_c=torch.stack(out,dim=2)
    return out_c
  
class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in anc
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, options = None,c_dim=175,dim=3,len=140,wid=140):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.len = len
        self.wid = wid

        # Submodules
        self.actvn = F.relu
        self.conv5 = nn.Conv2d(512,256,3,padding=1)
        self.conv4 = nn.Conv2d(256,128,3,padding=1)
        self.conv3 = nn.Conv2d(128,56,3,padding=1)
        self.conv2 = nn.Conv2d(56,56,3,padding=1)
        

        self.fc_1 = nn.Linear(self.len*self.wid, self.len*self.wid)
        self.fc_0 = nn.Linear(self.len*self.wid, self.len*self.wid)
        


    def forward(self, sig):

        # [Batch,Channel,Sample,Antenna]->[8,512,2,175] -> [8,256,2,175]
        out_5 = self.actvn(self.conv5(sig))
        
        #  [8,256,2,175] -> [8,128,2,175]
        out_4 = self.actvn(self.conv4(out_5))

        #  [8,128,2,175] -> [8,56,2,175]
        out_3 = self.actvn(self.conv3(out_4))

        #  [8,56,2,175] -> [8,56,2,175]
        out_2 = self.actvn(self.conv2(out_3))

        #  [8,56,2,175]-> [8,19600]
        out_1 = self.actvn(self.conv1(out_2))

        #  [8,19600]-> [8,19600]
        out_0 = self.fc_1(self.actvn(out_1))
        out_0 = self.fc_0(self.actvn(out_0))

        #  [8,19600] -> [8,140,140]
        out = out_0.view(-1,self.len,self.wid)

        return {
           "pred_proj": out
        }



class S2P_2views_conv_Model(nn.Module):
    '''
    Signal2PC Network that use the signal received from two views.
    
    The network has two encoder to extract feature from I signal and Q signal, then fusion the two part of feature
    by tensor add, then put the feature to the Decoder, Decoder output the predicted point cloud, shape is [B * 1024 * 3]
    
    '''
    def __init__(self, options = None):
        super().__init__()
        self.encoder = ComplexEncoder(options=options)
        self.decoder = Decoder(options=options)

   
    def forward(self,signals):
        proj_all=[]
               
        for signal in signals:
           out = self.encoder(signal['signal_real'],signal['signal_imag'])
           proj = self.decoder(out)
           proj_all.append(proj)
           
        feature_cat = torch.stack(proj_all,dim=2)
     
        return points

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # encoder = ConvEncoder()
  # decoder = Decoder()
  options = edict()
  options.se = True
  model = S2P_2views_conv_Model(options).to(device)
  signal_real = torch.randn(1, 9, 2800).to(device)
  signal_imag = torch.randn(1, 9, 2800).to(device)
  signal = {'signal_real':signal_real,'signal_imag':signal_imag}
  signals = []
  for i in range(4):
     signals.append(copy.deepcopy(signal))
  # summary(model, [(9,2800),(9,2800)])
  output = model(signals)
  print(output['pred_coord'].shape)