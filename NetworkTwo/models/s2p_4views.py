import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy
class SELayer(nn.Module):
    '''
        channel attention module, may be useless
    '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class ConvEncoder(nn.Module):
  '''
  Encoder same as the S2PModel encoder
  '''
  def __init__(self,options = None):
    self.options = options
    super(ConvEncoder,self).__init__()
    self.conv0 = nn.Conv1d(9,32,3,stride=3)
    self.conv1 = nn.Conv1d(32,64,3,stride=3)
    self.conv2 = nn.Conv1d(64,128,3,stride=3)
    self.conv3 = nn.Conv1d(128,256,3,stride=3)
    self.conv4 = nn.Conv1d(256,512,3,stride=3)
    self.conv5 = nn.Conv1d(512,1024,3,stride=3)
    self.conv6 = nn.Conv1d(1024,2048,3,stride=1)
    self.actvn = nn.ReLU()
    if options.se:
        # self.se0 = SELayer(9)
        # self.se1 = SELayer(32)     
        self.se2 = SELayer(64) 
        self.se3 = SELayer(128) 
        self.se4 = SELayer(256) 
        self.se5 = SELayer(512) 
        self.se6 = SELayer(1024)  

  def forward(self,net):
    batch_size = net.size(0)
    # if options.se:
    #     net = self.se0(net)
    net = self.conv0(net)
    net = self.actvn(net)
    # if options.se:
    #     net = self.se1(net)
    net = self.conv1(net)
    net = self.actvn(net)
    if self.options.se:
        net = self.se2(net)
    net = self.conv2(net)
    net = self.actvn(net)
    if self.options.se:
        net = self.se3(net)
    net = self.conv3(net)
    net = self.actvn(net)
    # if self.options.se:
    #     net = self.se4(net)
    net = self.conv4(net)
    net = self.actvn(net)
    # if self.options.se:
    #     net = self.se5(net)
    net = self.conv5(net)
    net = self.actvn(net)
    # if self.options.se:
    #     net = self.se6(net)
    net = self.conv6(net)
    net = self.actvn(net)

    return torch.flatten(net, start_dim=1)
  
class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self,options = None, dim=3, c_dim=2048, n_points=1024,):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points

        # Submodules
        self.actvn = F.relu
        self.fc_0 = nn.Linear(c_dim, 1024)
        self.fc_1 = nn.Linear(1024, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, dim*n_points)

    def forward(self, c):
        batch_size = c.size(0)

        net = self.fc_0(c)
        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(batch_size, self.n_points, self.dim)

        return {
           "pred_coord": points
        }


class S2P_4views_Model(nn.Module):
    '''
    Signal2PC Network that use the signal received from four views.
    
    The network has two encoder to extract feature from I signal and Q signal, then fusion the two part of feature
    by tensor add, then put the feature to the Decoder, Decoder output the predicted point cloud, shape is [B * 1024 * 3]
    
    '''
    def __init__(self, options = None):
        super().__init__()
        self.encoder_real = ConvEncoder(options=options)
        self.encoder_imag = ConvEncoder(options=options)
        self.decoder = Decoder(options=options,c_dim=2048*4)
   
    def forward(self,signals):
        features = []
        for signal in signals:
           c_real = self.encoder_real(signal['signal_real'])
           c_imag = self.encoder_imag(signal['signal_imag'])
           c = torch.add(c_real,c_imag)
           features.append(c)
        features = torch.concat(features,dim=1)
        points = self.decoder(features)
        
        return points

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # encoder = ConvEncoder()
  # decoder = Decoder()
  options = edict()
  options.se = False
#   model = S2P_4views_Model(options).to(device)
#   signal_real = torch.randn(1, 9, 2800).to(device)
#   signal_imag = torch.randn(1, 9, 2800).to(device)
#   signal = {'signal_real':signal_real,'signal_imag':signal_imag}
#   signals = []
#   for i in range(4):
#      signals.append(copy.deepcopy(signal))
  encoder = ConvEncoder(options=options).to(device=device)
  summary(encoder, (9,2800), 1)
  # summary(model, [(9,2800),(9,2800)])
#   output = model(signals)
#   print(output['pred_coord'].shape)