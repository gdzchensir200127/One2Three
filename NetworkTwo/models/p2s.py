import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from models.pointnet_utils import PointNetEncoder
from scipy.signal import gausspulse
import open3d as o3d
import time
class PointNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.pointEncoder = PointNetEncoder(global_feat=False,feature_transform=True,channel=3)
    self.conv1 = torch.nn.Conv1d(1088, 512, 1)
    self.conv2 = torch.nn.Conv1d(512, 256, 1)
    self.conv3 = torch.nn.Conv1d(256, 128, 1)
    self.conv4 = torch.nn.Conv1d(128, 32, 1)
    self.conv5 = torch.nn.Conv1d(32, 8, 1)
    self.conv6 = torch.nn.Conv1d(8, 1, 1)
    self.bn1 = nn.BatchNorm1d(512)
    self.bn2 = nn.BatchNorm1d(256)
    self.bn3 = nn.BatchNorm1d(128)
    self.bn4 = nn.BatchNorm1d(32)
    self.bn5 = nn.BatchNorm1d(8)
                                                                                                                                                                                                              

  def forward(self,x):
    '''
    x is the point cloud [B*3*N]
    '''
    batchsize = x.size()[0]
    n_pts = x.size()[2]
    x, trans, trans_feat = self.pointEncoder(x)  # x [B*1088*N]
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = F.relu(self.bn5(self.conv5(x)))
    x = self.conv6(x)

    return torch.squeeze(x)  # x [B*1*N]

class P2SModel(nn.Module):
  def __init__(self,options:None):
    super().__init__()
    self.options = options
    self.pn = PointNet()
    self.n_pts = 1024
    self.radar_pos = options.radar_pos
    # self.bins = options.bins
    self.radar_x = self.radar_pos[0]
    self.radar_y = self.radar_pos[1]
    self.radar_z = self.radar_pos[2]
    self.camera = [self.radar_x,self.radar_y,self.radar_z]
    self.radius = 50
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.start_bin = self.bins[0]
    # self.end_bin = self.bins[1]
    self.light_speed = 2.998e8
    self.fb = 1.4e9
    self.fc = 7.29e9
    self.bw = self.fb/self.fc
    self.sample_rate = 23.328e9
    self.decimate_factor = 8
    self.real_sample_rate = self.sample_rate/self.decimate_factor
    self.sample_interval = 1/self.real_sample_rate
    self.bin_num = options.bin_num
    # self.t_bin = np.array([b*self.sample_interval for b in range(self.start_bin,self.end_bin+1)]).astype(np.float32)
    # # self.t_bin = np.arange(self.start_bin*self.sample_interval,self.end_bin*self.sample_interval,self.sample_interval).astype(np.float32)
    # self.bin_num = self.t_bin.shape[0]
    # self.t_bin = np.reshape(self.t_bin,(1,self.t_bin.shape[0]))
    # self.t_bin = np.repeat(self.t_bin,self.n_pts,axis=0)
    self.FPS = 40   # 40Hz FPS
    self.moving_speed = 0.01 # 1cm/s
    self.moving_step = self.moving_speed/self.FPS
    self.x_start = -0.35
    self.x_end = 0.35
    self.steps = (int((self.x_end - self.x_start) / self.moving_step) + 1) // options.downsample_factor
    self.x_offsets = np.linspace(self.x_start,self.x_end,self.steps)
    self.offsets = np.zeros((self.steps,3),dtype=np.float32)
    self.offsets[:,0:1] = self.x_offsets.reshape(-1,1)   # 280 * 3
  
  def gauss(self,T_bin:np.array,Tp:torch.Tensor,batchsize,n_pts,start_bin,end_bin):
    '''
    Tp B*N
    '''
    # g = []
    # for i in range(batchsize):
    #   t_bin = np.array([b*self.sample_interval for b in range(start_bin[i],end_bin[i]+1)]).astype(np.float32)
    #   t_bin = np.reshape(t_bin,(1,t_bin.shape[0]))
    #   t_bin = np.repeat(t_bin,n_pts,axis=0) # N*bin
    #   t = t_bin - np.reshape(Tp.cpu().numpy()[i],(Tp.shape[1],1))
    #   signal = gausspulse(t,fc=self.fc,bw=self.bw,retquad=True,retenv=True)[2]
    #   g.append(signal)
    # signal = np.stack(g,axis=0)
    # t_bin = np.reshape(self.t_bin,(1,self.t_bin.shape[0],self.t_bin.shape[1]))
    # t_bin = np.repeat(t_bin,batchsize,axis=0)
    t = T_bin - np.reshape(Tp.cpu().numpy(),(Tp.shape[0],Tp.shape[1],1))
    signal = gausspulse(t,fc=self.fc,bw=self.bw,retquad=True,retenv=True)[2]
    return torch.from_numpy(signal/2)
  
  def get_visibility(self,pc:torch.Tensor):
    pc_np = pc.cpu().numpy()
    batch_size = pc.size()[0]
    n_pts = pc.size()[2]
    V = np.zeros((batch_size,n_pts)).astype(np.float32)
    for i in range(batch_size):
      points = pc_np[i,:,:]
      points = np.transpose(points,(1,0))
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(points)
      a,pt_map = pcd.hidden_point_removal(self.camera, self.radius)
      for index in pt_map:
        V[i,index] = 1
    V = torch.from_numpy(V)
    return V


  def forward(self,x:torch.Tensor, start_bin, end_bin):
    '''
    x is the point cloud [B*3*N] 
    '''
    x = torch.transpose(x,2,1)
    batchsize = x.size()[0]
    n_pts = x.size()[2]
    assert(self.n_pts == n_pts)
    P = self.pn(x).unsqueeze(2).repeat(1,1,self.bin_num)
    I = []
    Q = []
    V = None
    T_bin = []
    for i in range(batchsize):
      t_bin = np.array([b*self.sample_interval for b in range(start_bin[i],end_bin[i]+1)]).astype(np.float32)
      t_bin = np.reshape(t_bin,(1,t_bin.shape[0]))
      t_bin = np.repeat(t_bin,n_pts,axis=0) # N*bin
      T_bin.append(t_bin)
    T_bin = np.stack(T_bin,axis=0)

    for i in range(self.steps):
      offset = torch.tensor(self.offsets[i,:]).unsqueeze(0).unsqueeze(2).repeat(batchsize,1,n_pts).to(self.device)
      pc = x + offset
      if i%(self.steps//self.options.visibility_n) == 0:
        V = self.get_visibility(pc).unsqueeze(2).repeat(1,1,self.bin_num).to(device=self.device)
      # V = torch.ones(batchsize,n_pts,self.bin_num).to(device=device)
      radar_point = torch.tensor([self.radar_x,self.radar_y,self.radar_z])
      radar_point = radar_point.unsqueeze(0).unsqueeze(2)
      radar_point = radar_point.repeat(batchsize,1,n_pts).to(self.device)
      diff = pc - radar_point
      dist = torch.norm(diff,dim = 1)         # distance [B*N] 
      Tp = torch.mul(dist,2/self.light_speed) # TOF [B*N]
      g = self.gauss(T_bin=T_bin,Tp=Tp,batchsize=batchsize,n_pts=n_pts,start_bin=start_bin,end_bin=end_bin).to(device=self.device) # [B*N*bin]
      R_I = torch.sin(torch.mul(Tp,2*(np.pi)*self.fc)).unsqueeze(2).repeat(1,1,self.bin_num)
      R_I = torch.mul(g,R_I)
      R_I = torch.div(R_I,torch.square(dist).unsqueeze(2).repeat(1,1,self.bin_num)) # [B*N*bin]
      S_I = torch.mul(torch.mul(P,V),R_I)
      S_I = torch.sum(S_I,dim=1)
      R_Q = torch.cos(torch.mul(Tp,2*(np.pi)*self.fc)).unsqueeze(2).repeat(1,1,self.bin_num)
      R_Q = torch.mul(g,R_Q)
      R_Q = torch.div(R_Q,torch.square(dist).unsqueeze(2).repeat(1,1,self.bin_num)) # [B*N*bin]
      S_Q = torch.mul(torch.mul(P,V),R_Q)
      S_Q = torch.sum(S_Q,dim=1)
      I.append(S_I)
      Q.append(S_Q)

    I = torch.stack(I,dim = -1)
    Q = torch.stack(Q,dim = -1)
    return {
      'I':I,
      'Q':Q
    }


if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # encoder = ConvEncoder()
  # decoder = Decoder()
  radar_pos = (0.,0.5,0.60)
  bins = (4,12)
  model = P2SModel(radar_pos=radar_pos,device=device,bins=bins).to(device)
  pc = torch.randn(8,3,1024).to(device)
#   signal_real = torch.randn(1, 9, 2800).to(device)
#   signal_imag = torch.randn(1, 9, 2800).to(device)
  # summary(model, (3,1024),3)
  for i in range(8):
    start_time1 = time.time()
    S_I,S_Q = model(pc)
    S_I_GT = torch.randn(8,9,280).to(device)
    S_Q_GT = torch.randn(8,9,280).to(device) 
    loss = nn.L1Loss()
    out = loss(S_I,S_I_GT)
    out.backward()
    end_time1 = time.time()
    time1 = end_time1 - start_time1
    print("函数1的执行时间：", time1)
  pass
  

