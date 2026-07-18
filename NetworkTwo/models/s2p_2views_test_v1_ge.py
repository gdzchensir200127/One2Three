import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy


class ComplexEncoder(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None):
    self.options = options
    super(ConvEncoder,self).__init__()
    self.actvn = nn.ReLU()

    # self.conv11Real = nn.Conv2d(9,16,1)
    self.conv13Real = nn.Conv2d(9,16,3,padding=1)
    self.conv15Real = nn.Conv2d(9,16,5,padding=2)
    self.conv17Real = nn.Conv2d(9,16,7,padding=3)
    self.conv19Real = nn.Conv2d(9,16,9,padding=5)

    # self.conv11Imag = nn.Conv2d(9,16,1)
    self.conv13Imag = nn.Conv2d(9,16,3,padding=1)
    self.conv15Imag = nn.Conv2d(9,16,5,padding=2)
    self.conv17Imag = nn.Conv2d(9,16,7,padding=3)
    self.conv19Imag = nn.Conv2d(9,16,9,padding=5)

    # self.conv21Real = nn.Conv2d(64,128,1)
    self.conv23Real = nn.Conv2d(64,128,3,padding=1)
    self.conv25Real = nn.Conv2d(64,128,5,padding=2)
    self.conv27Real = nn.Conv2d(64,128,7,padding=3)
    self.conv29Real = nn.Conv2d(64,128,9,padding=5)

    # self.conv21Imag = nn.Conv2d(64,128,1)
    self.conv23Imag = nn.Conv2d(64,128,3,padding=1)
    self.conv25Imag = nn.Conv2d(64,128,5,padding=2)
    self.conv27Imag = nn.Conv2d(64,128,7,padding=3)
    self.conv297Imag = nn.Conv2d(64,128,9,padding=5)

    # self.conv31Real = nn.Conv2d(512,512,1)
    self.conv33Real = nn.Conv2d(512,512,3,padding=1)
    self.conv35Real = nn.Conv2d(512,512,5,padding=2)
    self.conv37Real = nn.Conv2d(512,512,7,padding=3)
    self.conv39Real = nn.Conv2d(512,512,9,padding=5)

    # self.conv31Imag = nn.Conv2d(512,512,1)
    self.conv33Imag = nn.Conv2d(512,512,3,padding=1)
    self.conv35Imag = nn.Conv2d(512,512,5,padding=2)
    self.conv37Imag = nn.Conv2d(512,512,7,padding=3)
    self.conv39Imag = nn.Conv2d(512,512,9,padding=5)
    

  def forward(self,real,imag):
    #Layer 1 (Channel: 9->16->64)
    # out_11_real = self.actvn(self.conv11Real(real)-self.conv11Imag(imag))
    # out_11_imag = self.actvn(self.conv11Imag(real)+self.conv11Real(imag))

    out_13_real = self.actvn(self.conv13Real(real)-self.conv13Imag(imag))
    out_13_imag = self.actvn(self.conv13Imag(real)+self.conv13Real(imag))

    out_15_real = self.actvn(self.conv15Real(real)-self.conv15Imag(imag))
    out_15_imag = self.actvn(self.conv15Imag(real)+self.conv15Real(imag))

    out_17_real = self.actvn(self.conv17Real(real)-self.conv17Imag(imag))
    out_17_imag = self.actvn(self.conv17Imag(real)+self.conv17Real(imag))
    
    out_19_real = self.actvn(self.conv19Real(real)-self.conv19Imag(imag))
    out_19_imag = self.actvn(self.conv19Imag(real)+self.conv19Real(imag))

    out_1_real = torch.concat((out_13_real,out_15_real,out_17_real,out_19_real),dim=1)
    out_1_imag = torch.concat((out_13_imag,out_15_imag,out_17_imag,out_19_imag),dim=1)  

    #Layer 2 (Channel: 64->128->512)
    # out_21_real = self.actvn(self.conv21Real(out_1_real)-self.conv21Imag(out_1_imag))
    # out_21_imag = self.actvn(self.conv21Imag(out_1_real)+self.conv21Real(out_1_imag))

    out_23_real = self.actvn(self.conv23Real(out_1_real)-self.conv23Imag(out_1_imag))
    out_23_imag = self.actvn(self.conv23Imag(out_1_real)+self.conv23Real(out_1_imag))

    out_25_real = self.actvn(self.conv25Real(out_1_real)-self.conv25Imag(out_1_imag))
    out_25_imag = self.actvn(self.conv25Imag(out_1_real)+self.conv25Real(out_1_imag))

    out_27_real = self.actvn(self.conv27Real(out_1_real)-self.conv27Imag(out_1_imag))
    out_27_imag = self.actvn(self.conv27Imag(out_1_real)+self.conv27Real(out_1_imag))

    out_29_real = self.actvn(self.conv29Real(real)-self.conv29Imag(imag))
    out_29_imag = self.actvn(self.conv29Imag(real)+self.conv29Real(imag))

    out_2_real = torch.concat((out_23_real,out_25_real,out_27_real,out_29_real),dim=1)
    out_2_imag = torch.concat((out_23_imag,out_25_imag,out_27_imag,out_29_imag),dim=1) 

    #Layer 3 (Channel: 512->512->512)
    # out_31_real = self.actvn(self.conv31Real(out_2_real)-self.conv31Imag(out_2_imag))
    # out_31_imag = self.actvn(self.conv31Imag(out_2_real)+self.conv31Real(out_2_imag))

    out_33_real = self.actvn(self.conv33Real(out_2_real)-self.conv33Imag(out_2_imag))
    out_33_imag = self.actvn(self.conv33Imag(out_2_real)+self.conv33Real(out_2_imag))

    out_35_real = self.actvn(self.conv35Real(out_2_real)-self.conv35Imag(out_2_imag))
    out_35_imag = self.actvn(self.conv35Imag(out_2_real)+self.conv35Real(out_2_imag))

    out_37_real = self.actvn(self.conv37Real(out_2_real)-self.conv37Imag(out_2_imag))
    out_37_imag = self.actvn(self.conv37Imag(out_2_real)+self.conv37Real(out_2_imag))

    out_39_real = self.actvn(self.conv39Real(out_2_real)-self.conv39Imag(out_2_imag))
    out_39_imag = self.actvn(self.conv39Imag(out_2_real)+self.conv39Real(out_2_imag))

    out_3_real = torch.add(out_33_real,out_35_real)
    out_3_real = torch.add(out_3_real,out_37_real)
    out_3_real = torch.add(out_3_real,out_39_real)
    out_3_imag = torch.add(out_33_imag,out_35_imag)
    out_3_imag = torch.add(out_3_imag,out_37_imag)
    out_3_imag = torch.add(out_3_imag,out_39_imag)

    return out_1_real, out_1_imag, out_2_real, out_2_imag, out_3_real, out_3_imag

class ConvEncoder(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None):
    self.options = options
    super(ConvEncoder,self).__init__()
    self.actvn = nn.ReLU()
    
    self.conv13 = nn.Conv1d(9,16,3,padding=1)
    self.conv15 = nn.Conv1d(9,16,5,padding=2)
    self.conv17 = nn.Conv1d(9,16,7,padding=3)
    self.conv19 = nn.Conv1d(9,16,9,padding=4)
    
    self.conv23 = nn.Conv1d(64,64,3,padding=1)
    self.conv25 = nn.Conv1d(64,64,5,padding=2)
    self.conv27 = nn.Conv1d(64,64,7,padding=3)
    self.conv29 = nn.Conv1d(64,64,9,padding=4)

    self.conv33 = nn.Conv1d(256,256,3,padding=1)
    self.conv35 = nn.Conv1d(256,256,5,padding=2)
    self.conv37 = nn.Conv1d(256,256,7,padding=3)
    self.conv39 = nn.Conv1d(256,256,9,padding=4)
    

  def forward(self,sig):
    #Layer 1 (Channel: 9->16->64)
    out_13 = self.actvn(self.conv13(sig))
    out_15 = self.actvn(self.conv15(sig))
    out_17 = self.actvn(self.conv17(sig))
    out_19 = self.actvn(self.conv19(sig))

    out_L1 = torch.concat((out_13,out_15,out_17,out_19),dim=1)  

    #Layer 2 (Channel: 64->64->256)
    out_23 = self.actvn(self.conv23(out_L1))
    out_25 = self.actvn(self.conv25(out_L1))
    out_27 = self.actvn(self.conv27(out_L1))
    out_29 = self.actvn(self.conv29(out_L1))

    out_L2 = torch.concat((out_23,out_25,out_27,out_29),dim=1) 

    #Layer 3 (Channel: 256->256->1024)
    out_33 = self.actvn(self.conv33(out_L2))
    out_35 = self.actvn(self.conv35(out_L2))
    out_37 = self.actvn(self.conv37(out_L2))
    out_39 = self.actvn(self.conv39(out_L2))

    out_L3 = torch.concat((out_33,out_35,out_37,out_39),dim=1)
    # out_L3 = torch.add(out_33,out_35)
    # out_L3 = torch.add(out_L3,out_37)
    # out_L3 = torch.add(out_L3,out_39)  

    return out_L1, out_L2, out_L3
  
class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, options = None,c_dim=2800,dim=3,n_points=1024,view=4,):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points
        self.view = view

        # Submodules
        self.actvn = F.relu
        self.conv3_real = nn.Conv2d(self.view,self.view,3,padding=1)
        self.conv3_imag = nn.Conv2d(self.view,self.view,3,padding=1)
        self.pool3_real = nn.MaxPool2d((4,1))
        self.pool3_imag = nn.MaxPool2d((4,1))

        self.conv2_real = nn.Conv2d(self.view,self.view,3,padding=1)
        self.conv2_imag = nn.Conv2d(self.view,self.view,3,padding=1)
        self.pool2_real = nn.MaxPool2d((8,1))
        self.pool2_imag = nn.MaxPool2d((8,1))

        self.conv1_real = nn.Conv2d(self.view,self.view,3,padding=1)
        self.conv1_imag = nn.Conv2d(self.view,self.view,3,padding=1)
        self.pool1_real = nn.MaxPool2d((16,8))
        self.pool1_imag = nn.MaxPool2d((16,8))

        self.conv0_real = nn.Conv2d(self.view,self.view,3,padding=1)
        self.conv0_imag = nn.Conv2d(self.view,self.view,3,padding=1)

        self.fc_0 = nn.Linear(self.view*self.c_dim, 2048)
        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, dim*n_points)


    def forward(self, sig_1_real, sig_1_imag, sig_2_real, sig_2_imag, sig_3_real, sig_3_imag):
        batch_size,_,_,_= sig_1_real.shape

        # [Batch,View,Sample,Antenna]->[8,View,1024,2800] -> [8,View,256,2800]
        out_3_real = self.actvn(self.conv3_real(sig_3_real)-self.conv3_imag(sig_3_imag))
        out_3_imag = self.actvn(self.conv3_imag(sig_3_imag)+self.conv3_real(sig_3_real))
        out_3_real = self.pool3_real(out_3_real)
        out_3_imag = self.pool3_imag(out_3_imag)

        # [8,View,256,2800] -> [8,View,512,2800] -> [8,View,64,2800]
        in_2_real = torch.concat((sig_2_real,out_3_real),dim=2)
        in_2_imag = torch.concat((sig_2_imag,out_3_imag),dim=2)
        out_2_real = self.actvn(self.conv2_real(in_2_real)-self.conv2_imag(in_2_imag))
        out_2_imag = self.actvn(self.conv2_imag(in_2_imag)+self.conv2_real(in_2_real))
        out_2_real = self.pool2_real(out_2_real)
        out_2_imag = self.pool2_imag(out_2_imag)

        # [[8,View,64,2800] -> [8,View,128,2800] -> [8,View,8,350]
        in_2_real = torch.concat((sig_1_real,out_2_real),dim=2)
        in_2_imag = torch.concat((sig_1_imag,out_2_imag),dim=2)
        out_1_real = self.actvn(self.conv1_real(sig_1_real)-self.conv1_imag(sig_1_imag))
        out_1_imag = self.actvn(self.conv1_imag(sig_1_imag)+self.conv1_real(sig_1_real))
        out_1_real = self.pool1_real(out_1_real)
        out_1_imag = self.pool1_imag(out_1_imag)

        # [8,View,8,350] -> [8,View,8,350] -> [8,View*2800] 
        out_0_real = self.actvn(self.conv0_real(out_1_real)-self.conv0_imag(out_1_imag))
        out_0_imag = self.actvn(self.conv0_imag(out_1_imag)+self.conv0_real(out_1_real))
        out_0 = torch.concat((out_0_real,out_0_imag),dim=1).view(batch_size,-1)

        out = self.fc_0(out_0)
        out = self.fc_1(self.actvn(out))
        out = self.fc_2(self.actvn(out))
        points = self.fc_out(self.actvn(out))
        points = points.view(batch_size, self.n_points, self.dim)

        return {
           "pred_coord": points
        }

class Construct(nn.Module):
    r''' Simple Construct for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self,options = None, dim=3, c_dim=44800, n_points=1024,):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points

        # Submodules
        self.actvn = F.relu
        self.fc_real = nn.Linear(c_dim, 1024)
        self.fc_imag = nn.Linear(c_dim, 1024)
        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, dim*n_points)

    def forward(self, sig1,sig2):
        batch_size = sig1.size(0)

        net_real = self.fc_real(sig1)
        net_imag = self.fc_real(sig2)
        net = torch.concat((net_real,net_imag),dim=1)
        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(batch_size, self.n_points, self.dim)

        return {
           "pred_coord": points
        }

class S2P_2views_test_Model(nn.Module):
    '''
    Signal2PC Network that use the signal received from two views.
    
    The network has two encoder to extract feature from I signal and Q signal, then fusion the two part of feature
    by tensor add, then put the feature to the Decoder, Decoder output the predicted point cloud, shape is [B * 1024 * 3]
    
    '''
    def __init__(self, options = None):
        super().__init__()
        # self.encoder = ComplexEncoder(options=options)
        self.encoder_real = ConvEncoder(options=options)
        self.encoder_imag = ConvEncoder(options=options)
        self.decoder = Decoder(options=options)
        self.construct = Construct(options=options)
   
    def forward(self,signals):
        feature_L1_real = []
        feature_L1_imag = []
        feature_L2_real = []
        feature_L2_imag = []
        feature_L3_real = []
        feature_L3_imag = []
        
        for signal in signals:
           out_1_real, out_2_real, out_3_real = self.encoder_real(signal['signal_real'])
           out_1_imag, out_2_imag, out_3_imag = self.encoder_imag(signal['signal_imag'])
           
           feature_L1_real.append(out_1_real)
           feature_L1_imag.append(out_1_imag)
           feature_L2_real.append(out_2_real)
           feature_L2_imag.append(out_2_imag)
           feature_L3_real.append(out_3_real)
           feature_L3_imag.append(out_3_imag)

        features_L1_real = torch.stack(feature_L1_real,dim=1)
        features_L1_imag = torch.stack(feature_L1_imag,dim=1)
        features_L2_real = torch.stack(feature_L2_real,dim=1)
        features_L2_imag = torch.stack(feature_L2_imag,dim=1)
        features_L3_real = torch.stack(feature_L3_real,dim=1)
        features_L3_imag = torch.stack(feature_L3_imag,dim=1)

        points=self.decoder(features_L1_real, features_L1_imag, features_L2_real, features_L2_imag, features_L3_real, features_L3_imag)      
        return points

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # encoder = ConvEncoder()
  # decoder = Decoder()
  options = edict()
  options.se = True
  model = S2P_2views_test_Model(options).to(device)
  signal_real = torch.randn(1, 9, 2800).to(device)
  signal_imag = torch.randn(1, 9, 2800).to(device)
  signal = {'signal_real':signal_real,'signal_imag':signal_imag}
  signals = []
  for i in range(4):
     signals.append(copy.deepcopy(signal))
  # summary(model, [(9,2800),(9,2800)])
  output = model(signals)
  print(output['pred_coord'].shape)