import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy
class ConvLayer(nn.Module):
  def __init__(self,options = None):
    super(ConvLayer,self).__init__()
    self.conv = nn.Conv1d(9,512,14,stride=14)
    self.actvn = nn.ReLU()

  def forward(self,x):
     x = self.conv(x)
     x = self.actvn(x)
     return x

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

class Transformer_Encoder(nn.Module):
    '''
    使用Transformer作为Encoder，做法类似VIT，通过卷积对每一个视角收到的信号进行切片，每个视角可以得到200个token，四个
    视角concat后有800个token，每个token的特征向量维度为1024，输入到Transformer Encoder Layer

    关于I路信号和Q路信号特征的融合，是通过分别卷积后再concat来进行的
    '''
    def __init__(self, options = None, num_patches = 800, embed_dim=1024):
      super().__init__()
      self.conv_real = ConvLayer()
      self.conv_imag = ConvLayer()
      self.feature_token = nn.Parameter(torch.randn(1, 1, embed_dim))
      self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
      nn.init.trunc_normal_(self.pos_embed, std=0.02)
      nn.init.trunc_normal_(self.feature_token, std=0.02)
      self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=8,batch_first=True)
      self.trans_encoder = nn.TransformerEncoder(encoder_layer=self.trans_encoder_layer,num_layers=options.trans_layers)
      # self.avg_pool = nn.AdaptiveAvgPool1d()

    def forward(self,signals):
      features = []
      for signal in signals:
        c_real = signal['signal_real']
        c_imag = signal['signal_imag']
        x_real = self.conv_real(c_real)
        x_imag = self.conv_imag(c_imag)
        x = torch.cat((x_real,x_imag),dim=1)
        x = x.transpose(1,2)   # [batch,token,dim]
        features.append(x)
      features = torch.cat(features,1)
      feature_token = self.feature_token.expand(features.shape[0],-1,-1)
      features = torch.cat((feature_token,features),dim=1)
      features = features + self.pos_embed
      features = self.trans_encoder(features)
      return features[:,0]
    
class S2P_4views_Transfomer(nn.Module):
    '''
    同样是使用四个视角的信号作为输入的Signal2PC网络
    
    使用Transformer作为Encoder，Decoder沿用之前的结构
    '''
    def __init__(self, options = None):
        super().__init__()
        self.encoder = Transformer_Encoder(options=options)
        self.decoder = Decoder(options=options,c_dim=1024)
   
    def forward(self,signals):
        features = self.encoder(signals)
        points = self.decoder(features)
        return points
    

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  options = edict()
  options.trans_layers = 3
  # options.se = True
  # model = S2P_4views_Model(options).to(device)
  # model = nn.TransformerEncoderLayer(d_model=252,nhead=7,batch_first=True).to(device)
  # model = ConvLayer().to(device)
  # input = torch.randn(1,9,2800).to(device)
  # output = model(input)
  # print(output.shape)
  # summary(model,input_size=(9,2800),batch_size=8)
  # summary(model,(200,252))
  # input = torch.randn(10,200,252).to(device)
  # output = model(input)
  # print(output.shape)
  signal_real = torch.randn(1, 9, 2800).to(device)
  signal_imag = torch.randn(1, 9, 2800).to(device)
  signal = {'signal_real':signal_real,'signal_imag':signal_imag}
  signals = []
  for i in range(4):
     signals.append(copy.deepcopy(signal))
  model = S2P_4views_Transfomer(options=options).to(device)
  output = model(signals)
  print(output['pred_coord'].shape)
  # summary(model, [(9,2800),(9,2800)])

#   output = model(signals)
#   print(output['pred_coord'].shape)