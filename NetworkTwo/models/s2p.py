import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
class ConvEncoder(nn.Module):
  '''
    Simple encoder for the Signal2PC Network

    The encoder consists of 6 layers Conv1d, the input is the signal of I or Q, shape is
    [B * 9(bins) * 2800(sample point)], output is the feature, shape is [B * 2048]

  '''
  def __init__(self):
    super().__init__()
    self.conv0 = nn.Conv1d(9,32,3,stride=3)
    self.conv1 = nn.Conv1d(32,64,3,stride=3)
    self.conv2 = nn.Conv1d(64,128,3,stride=3)
    self.conv3 = nn.Conv1d(128,256,3,stride=3)
    self.conv4 = nn.Conv1d(256,512,3,stride=3)
    self.conv5 = nn.Conv1d(512,1024,3,stride=3)
    self.conv6 = nn.Conv1d(1024,2048,3,stride=1)
    self.actvn = nn.ReLU()

  def forward(self,x):
    batch_size = x.size(0)
    net = self.conv0(x)
    net = self.conv1(self.actvn(net))
    net = self.conv2(self.actvn(net))
    net = self.conv3(self.actvn(net))
    net = self.conv4(self.actvn(net))
    net = self.conv5(self.actvn(net))
    net = self.conv6(self.actvn(net))

    return torch.flatten(net, start_dim=1)
  
class Decoder(nn.Module):
    r''' Simple decoder for the Signal2PC Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, dim=3, c_dim=2048, n_points=1024):
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


class S2PModel(nn.Module):
    '''
    Signal2PC Network that use the signal received from the single view.
    
    The network has two encoder to extract feature from I signal and Q signal, then fusion the two part of feature
    by tensor add, then put the feature to the Decoder, Decoder output the predicted point cloud, shape is [B * 1024 * 3]
    '''
    def __init__(self, option = None, encoder_real = ConvEncoder(),encoder_imag = ConvEncoder(),decoder = Decoder()):
        super().__init__()
        self.encoder_real = encoder_real
        self.encoder_imag = encoder_imag 
        self.decoder = decoder   
   

    def forward(self, x_real,x_imag):
        c_real = self.encoder_real(x_real)
        c_imag = self.encoder_imag(x_imag)
        c = torch.add(c_real,c_imag)
        points = self.decoder(c)
        return points

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # encoder = ConvEncoder()
  # decoder = Decoder()
  model = S2PModel().to(device)
  signal_real = torch.randn(1, 9, 2800).to(device)
  signal_imag = torch.randn(1, 9, 2800).to(device)
  summary(model, [(9,2800),(9,2800)])
  output = model(signal_real,signal_imag)
  print(output.shape)