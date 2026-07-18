from signal_simulate.util.mesh_cp import TriMesh
from signal_simulate.util.plot import plot_radar_data,plot_iq_data
import math
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import warnings
import copy
from scipy.io import savemat
import time
from multiprocessing import Process, Queue, Lock
import random,sys
from scipy.signal import gausspulse

# radar parameters
fb = 1.4e9
fc = 7.29e9
bw =fb/fc

sample_rate = 23.328e9

light_speed = 2.998e8

decimate_factor = 8

real_sample_rate = sample_rate/decimate_factor

sample_interval = 1/real_sample_rate

bins = 192

# trans signal
# tao = (2*math.pi*fb*(math.log10(math.e))**(1/2))**(-1)

# g = lambda t:cp.exp(-(cp.square(t))/(2*(tao**2)))

g = lambda t:gausspulse(t,fc=fc,bw=bw,retquad=True,retenv=True)[2]

cos_carrier = lambda t:cp.cos(2*cp.pi*fc*t)
sin_carrier = lambda t:cp.sin(2*cp.pi*fc*t)

# sample time

t = cp.arange(0,bins*sample_interval,sample_interval)

sample_point = t.size

def frame_simulate(meshs:list, radar_pos:cp.ndarray) -> cp.ndarray:
  '''
  simulate the one frame of the UWB radar signal reflected by the object
  param mesh: the TriMesh of the object
  param radar_pos: the position numpy array of the radar [x,y,z]
  '''

  vert1 = cp.zeros((len(meshs),meshs[0].faces.shape[0],3),dtype=cp.float64)
  vert2 = cp.zeros((len(meshs),meshs[0].faces.shape[0],3),dtype=cp.float64)
  vert3 = cp.zeros((len(meshs),meshs[0].faces.shape[0],3),dtype=cp.float64)

  for  index,mesh in enumerate(meshs):
    faces = mesh.faces
    verts = mesh.verts.tolist()
    Vert1 = cp.array([verts[i-1] for i in faces[:,0].tolist()])
    Vert2 = cp.array([verts[i-1] for i in faces[:,1].tolist()])
    Vert3 = cp.array([verts[i-1] for i in faces[:,2].tolist()])
    vert1[index,:,:] = Vert1
    vert2[index,:,:] = Vert2
    vert3[index,:,:] = Vert3
  vec1 = vert2-vert1
  vec2 = vert3-vert1

  normal_vec = cp.cross(vec1,vec2)

  normal_vec_norm = cp.linalg.norm(normal_vec,axis=2)

  mask = normal_vec_norm != 0

  if not cp.all(mask):
    vert1 = vert1[mask]
    vert2 = vert2[mask]
    vert3 = vert3[mask]
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    normal_vec = normal_vec[mask]
    normal_vec_norm = normal_vec_norm[mask]

  midpoint = (vert1+vert2+vert3)/3
  signal_vec = midpoint - radar_pos
  signal_vec_norm = cp.linalg.norm(signal_vec,axis=2)
  
  dot_product = cp.sum(normal_vec*signal_vec,axis=2)

  # dot_product = cp.einsum('ij,ij->i',normal_vec,signal_vec)
  angle = cp.arccos(dot_product/(normal_vec_norm*signal_vec_norm))
  
  theta = cp.where(angle<cp.pi/2 ,angle ,cp.pi-angle)*2
  R = cp.linalg.norm(signal_vec,axis=2)  # distance
  Tp = 2*R/light_speed  # TOF
  Tp = Tp[cp.newaxis,:,:]
  Tp = cp.repeat(Tp,bins,axis=0)
  global t
  t = t[:,cp.newaxis,cp.newaxis]
  # Q component
  Q = (1/2) * g(t-Tp) * cos_carrier(Tp)
  # I component
  I = (1/2) * g(t-Tp) * sin_carrier(Tp)
  # signal strength
  sigma_square = 40
  orientation_factor = cp.exp(-(theta**2)/(2*sigma_square))
  area_factor =  0.5 * normal_vec_norm
  distance_factor = 1/(R**2)
  strength = orientation_factor*area_factor*distance_factor
  strength = strength[cp.newaxis,:]
  signal = (I+Q*1j)*strength

  receive_signal = cp.sum(signal,axis=2)

  return receive_signal

def object_move_signal_simulate(init_mesh:TriMesh,radar_pos:tuple,offsets:cp.ndarray) -> cp.ndarray:
  '''
  simulate the signal of a move object
  : param init_mesh: the init TiiMesh of the object
  : param radar_pos: the position numpy array of the radar [x,y,z]
  : param offsets: the position offset numpy array of the moving object, a [N * 3] array, N is the moving steps,
  each row is the position offset of each move step, such as [[x_offset1,y_offset1,z_offset1],...] 
  '''
  steps = offsets.shape[0]
  signal = cp.zeros((steps,sample_point),dtype=cp.complex128)
  meshs = []
  for i in range(steps):
    offset = offsets[i]
    mesh = copy.deepcopy(init_mesh)
    mesh.verts += offset
    meshs.append(mesh)
  
  signal = frame_simulate(meshs=meshs,radar_pos=radar_pos)

  return signal


if __name__ == '__main__':

  # mesh = TriMesh.load_from_obj('/home/muxin/hdd/signal_simulate/data/mesh/cylinder.obj')
  mesh = TriMesh.load_from_obj('/home/muxin/hdd/signal_simulate/data/mesh/display_after_HPR_decimate.obj')

  radar_pos = cp.array([0,0.60,0.08])


  FPS = 40   # 40Hz FPS
  moving_speed = 0.5 # 1cm/s
  moving_step = moving_speed/FPS
  x_start = -0.35
  x_end = 0.35
  steps = int((x_end - x_start) / moving_step) + 1
  x_offsets = cp.linspace(x_start,x_end,steps)
  offsets = cp.zeros((steps,3),dtype=float)
  offsets[:,0:1] = x_offsets.reshape(-1,1)

  # y_start = 0
  # y_end = -0.1
  # steps = int(np.abs(y_end - y_start) / moving_step) + 1
  # y_offsets = np.linspace(y_start,y_end,steps)
  # offsets = np.zeros((steps,3),dtype=float)
  # offsets[:,1:2] = y_offsets.reshape(-1,1)
  start_time = time.time()

  signal = cp.asnumpy(object_move_signal_simulate(init_mesh=mesh,radar_pos=radar_pos,offsets=offsets))

  end_time = time.time()
  execution_time = end_time - start_time
  print(f"simulate time:{execution_time} s")


  savemat('/home/muxin/hdd/signal_simulate/data/simulate_data/display_60_cp.mat',{'data':signal})

  fig,ax = plt.subplots(1,1)

  plot_radar_data(ax,signal,'simulate data')

  plt.show()

  print(f'all processes waited')






