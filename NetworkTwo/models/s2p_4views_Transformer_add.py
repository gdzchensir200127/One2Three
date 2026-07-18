import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy
class ConvLayer(nn.Module):
  def __init__(self,options = None):
    super(ConvLayer,self).__init__()
    self.conv = nn.Conv1d(9,1024,14,stride=14)
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
    对之前的Transformer Encoder进行修改，I路信号和Q路信号特征的融合通过分别卷积后add来进行
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
        x = torch.add(x_real,x_imag)
        x = x.transpose(1,2)   # [batch,token,dim]
        features.append(x)
      features = torch.cat(features,1)
      feature_token = self.feature_token.expand(features.shape[0],-1,-1)
      features = torch.cat((feature_token,features),dim=1)
      features = features + self.pos_embed
      features = self.trans_encoder(features)
      return features[:,0]
    
class Transformer_Encoder_2(nn.Module):
    '''
    这种Transformer Encoder只关注一个视角的信号，每个视角的信号可以得到200个token，特征维度为1024，将200个token输入Transformer Encoder，
    得到1024维度的特征，四个视角的信号可以得到四个特征，通过concat得到1024*4的特征，四个视角共享同一个Transformer Encoder，I路信号和Q路信号
    的特征融合同样是通过add来进行
    '''
    def __init__(self, options = None, num_patches = 200, embed_dim=1024):
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
        x = torch.add(x_real,x_imag)
        x = x.transpose(1,2)   # [batch,token,dim]
        feature_token = self.feature_token.expand(x.shape[0],-1,-1)
        x = torch.cat((feature_token,x),dim=1)
        x = x + self.pos_embed
        x = self.trans_encoder(x)
        features.append(x[:,0])
      features = torch.cat(features,1)
      return features
    
class S2P_4views_Transfomer_add(nn.Module):
    '''
    通过设置参数可以选择使用的Encoder，来决定Transformer是仅在单视角信号中进行自注意力机制，还是在四个视角的信号中进行自注意力机制
    实验表明，在单视角信号中进行自注意机制，之后再通过concat的方式拼接特征的方式效果比较好
    '''
    def __init__(self, options = None):
        super().__init__()
        if options.trans_atten_in_single_view:
          self.encoder = Transformer_Encoder_2(options=options)
          self.decoder = Decoder(options=options,c_dim=1024*4)      
        else:
          self.encoder = Transformer_Encoder(options=options)
          self.decoder = Decoder(options=options,c_dim=1024)
   
    def forward(self,signals):
        features = self.encoder(signals)
        points = self.decoder(features)
        return points
    

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  options = edict()
  options.trans_layers = 6
  options.trans_atten_in_single_view = True
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
  model = S2P_4views_Transfomer_add(options=options).to(device)
  output = model(signals)
  print(output['pred_coord'].shape)
  # summary(model, [(9,2800),(9,2800)])

#   output = model(signals)
#   print(output['pred_coord'].shape)