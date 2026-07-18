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
    
    self.conv3 = nn.Conv1d(self.in_ch,self.out_ch,3,padding=1)
    self.conv5 = nn.Conv1d(self.in_ch,self.out_ch,5,padding=2)
    self.conv7 = nn.Conv1d(self.in_ch,self.out_ch,7,padding=3)
    self.conv9 = nn.Conv1d(self.in_ch,self.out_ch,9,padding=4)
    

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

    #Layer 2 : -> [8,64,700]
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

    out_1=[]
    out_1.append(out_1_real)
    out_1.append(out_1_imag)
    out_1_c=torch.stack(out_1,dim=2)

    out_2=[]
    out_2.append(out_2_real)
    out_2.append(out_2_imag)
    out_2_c=torch.stack(out_2,dim=2)

    out_3=[]
    out_3.append(out_3_real)
    out_3.append(out_3_imag)
    out_3_c=torch.stack(out_3,dim=2)

    out_4=[]
    out_4.append(out_4_real)
    out_4.append(out_4_imag)
    out_4_c=torch.stack(out_4,dim=2)

    out_5=[]
    out_5.append(out_5_real)
    out_5.append(out_5_imag)
    out_5_c=torch.stack(out_5,dim=2)
    return out_1_c,out_2_c,out_3_c,out_4_c,out_5_c
  
class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in anc
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, options = None,c_dim=175,dim=3,n_points=1024,view=4,):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points
        self.view = view

        # Submodules
        self.actvn = F.relu
        self.conv5 = nn.Conv2d(512,256,3,padding=1)
        self.conv4 = nn.Conv2d(256,128,3,padding=1)
        self.conv3 = nn.Conv2d(128,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,32,3,padding=1)
        self.conv1 = nn.Conv2d(32,8,3,padding=1)
        
        # Pooling
        self.pool3 = nn.AvgPool2d((1,2))
        self.pool2 = nn.AvgPool2d((1,4))
        self.pool1 = nn.AvgPool2d((1,8))
        self.pool0 = nn.AvgPool2d((5,4))

        self.fc_0 = nn.Linear(2*2*self.view*self.c_dim, 1024)
        self.fc_1 = nn.Linear(1024, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, dim*n_points)


    def forward(self, sig_1, sig_2, sig_3, sig_4, sig_5):
        batch_size,_,_,_= sig_1.shape

        # [Batch,Channel,Sample,Antenna]->[8,512,2,175*view] -> [8,256,2,700]
        out_5 = self.actvn(self.conv5(sig_5))
        
        #  [8,256,2,700] cat [8,256,2,700] -> [8,256,4,700]-> [8,128,4,700]
        in_4 = torch.concat((sig_4,out_5),dim=2)
        out_4 = self.actvn(self.conv4(in_4))

        #  [8,128,4,700] cat [8,128,2,700] -> [8,128,6,700]-> [8,64,6,700]
        in_3 = torch.concat((self.pool3(sig_3),out_4),dim=2)
        out_3 = self.actvn(self.conv3(in_3))

        #  [8,64,6,700] cat [8,64,2,700] -> [8,64,8,700]-> [8,32,8,700]
        in_2 = torch.concat((self.pool2(sig_2),out_3),dim=2)
        out_2 = self.actvn(self.conv2(in_2))

        #  [8,32,8,700] cat [8,32,2,700] -> [8,32,10,700]-> [8,8,10,700]
        in_1 = torch.concat((self.pool1(sig_1),out_2),dim=2)
        out_1 = self.actvn(self.conv1(in_1))

        #  [8,8,10,700] -> [8,8,2,175]-> [8,2800]
        out_0 = (self.pool0(out_1)).view(batch_size,-1)

        out = self.fc_0(out_0)
        out = self.fc_1(self.actvn(out))
        out = self.fc_2(self.actvn(out))
        points = self.fc_out(self.actvn(out))
        points = points.view(batch_size, self.n_points, self.dim)

        return {
           "pred_coord": points
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
        # self.dim=0
   
    def forward(self,signals):
        feature_L1=[]
        feature_L2=[]
        feature_L3=[]
        feature_L4=[]
        feature_L5=[]
               
        for signal in signals:
           out_1,out_2,out_3,out_4,out_5 = self.encoder(signal['signal_real'],signal['signal_imag'])
           feature_L1.append(out_1)
           feature_L2.append(out_2)
           feature_L3.append(out_3)
           feature_L4.append(out_4)
           feature_L5.append(out_5)

        feature_L1_cat = torch.concat(feature_L1,dim=3)
        feature_L2_cat = torch.concat(feature_L2,dim=3)
        feature_L3_cat = torch.concat(feature_L3,dim=3)
        feature_L4_cat = torch.concat(feature_L4,dim=3)
        feature_L5_cat = torch.concat(feature_L5,dim=3)


       
        # points=self.decoder(features_L1_real, features_L1_imag, features_L2_real, features_L2_imag, features_L3_real, features_L3_imag, features_L4_real, features_L4_imag, features_L5_real, features_L5_imag) 
        points=self.decoder(feature_L1_cat,feature_L2_cat,feature_L3_cat,feature_L4_cat,feature_L5_cat)     
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