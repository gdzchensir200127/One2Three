from signal_simulate.util.mesh_cp import TriMesh
from signal_simulate.util.plot import plot_radar_data,plot_iq_data
import math
import matplotlib.pyplot as plt
import cupy as cp
import warnings
import copy
from scipy.io import savemat
import time
from scipy.signal import gausspulse

warnings.filterwarnings('error', category=cp.ComplexWarning)
warnings.filterwarnings('error', category=RuntimeWarning)

# radar parameters
fb = 1.4e9
fc = 7.29e9
bw =fb/fc

sample_rate = 23.328e9

light_speed = 2.998e8

decimate_factor = 8

real_sample_rate = sample_rate/decimate_factor

sample_interval = 1/real_sample_rate

bins = 30

# trans signal
# tao = (2*math.pi*fb*(math.log10(math.e))**(1/2))**(-1)

# g = lambda t:cp.exp(-(cp.square(t))/(2*(tao**2)))

g = lambda t:gausspulse(t,fc=fc,bw=bw,retquad=True,retenv=True)[2]

cos_carrier = lambda t:cp.cos(2*cp.pi*fc*t)
sin_carrier = lambda t:cp.sin(2*cp.pi*fc*t)

# sample time

t = cp.arange(0,bins*sample_interval,sample_interval)

sample_point = t.size


def frame_simulate(mesh:TriMesh, radar_pos:cp.ndarray) -> cp.ndarray:
  '''
  simulate the one frame of the UWB radar signal reflected by the object
  param mesh: the TriMesh of the object
  param radar_pos: the position numpy array of the radar [x,y,z]
  '''
  faces = mesh.faces
  verts = mesh.verts.tolist()
  vert1 = cp.array([verts[i-1] for i in faces[:,0].tolist()])
  vert2 = cp.array([verts[i-1] for i in faces[:,1].tolist()])
  vert3 = cp.array([verts[i-1] for i in faces[:,2].tolist()])
  vec1 = vert2-vert1
  vec2 = vert3-vert1
  normal_vec = cp.cross(vec1, vec2)
  normal_vec_norm = cp.linalg.norm(normal_vec,axis=1)

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
  signal_vec_norm = cp.linalg.norm(signal_vec,axis=1)

  dot_product = cp.sum(normal_vec*signal_vec,axis=1)
  angle = cp.arccos(dot_product/(normal_vec_norm*signal_vec_norm))
  
  theta = cp.where(angle<cp.pi/2 ,angle ,cp.pi-angle)*2
  R = cp.linalg.norm(signal_vec,axis=1)  # distance
  Tp = 2*R/light_speed  # TOF
  Tp = cp.repeat(Tp.reshape(-1,1),bins,axis=1)
  t = cp.arange(0,bins*sample_interval,sample_interval)

  sample_point = t.size
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
  signal = (I+Q*1j)*strength.reshape(-1,1)

  receive_signal = cp.sum(signal,axis=0)

  return receive_signal


def object_move_signal_simulate(init_mesh:TriMesh,radar_pos:cp.ndarray,offsets:cp.ndarray) -> cp.ndarray:
  '''
  simulate the signal of a move object
  : param init_mesh: the init TiiMesh of the object
  : param radar_pos: the position numpy array of the radar [x,y,z]
  : param offsets: the position offset numpy array of the moving object, a [N * 3] array, N is the moving steps,
  each row is the position offset of each move step, such as [[x_offset1,y_offset1,z_offset1],...] 
  '''
  steps = offsets.shape[0]
  signal = cp.zeros((steps,sample_point),dtype=cp.complex128)
  for i in range(steps):
    offset = offsets[i]
    mesh = copy.deepcopy(init_mesh)
    mesh.verts += offset
    signal[i:,] = frame_simulate(mesh=mesh,radar_pos=radar_pos)
    pass

  return signal
  
def simulate(input_path:str,output_path:str,display:bool = False):
  radar_pos = cp.array([0,0.74,0.1])
  mesh = TriMesh.load_from_obj(input_path)
  
  # if mesh.faces.shape[0] > 15000:
  #   print("file {} have faces more than 15000, skip".format(input_path))
  #   return

  start_time = time.time()
  FPS = 40   # 40Hz FPS
  moving_speed = 0.01 # 1cm/s
  moving_step = moving_speed/FPS
  x_start = -0.35
  x_end = 0.35
  steps = int((x_end - x_start) / moving_step) + 1
  x_offsets = cp.linspace(x_start,x_end,steps)
  offsets = cp.zeros((steps,3),dtype=float)
  offsets[:,0:1] = x_offsets.reshape(-1,1)

  signal = cp.asnumpy(object_move_signal_simulate(init_mesh=mesh,radar_pos=radar_pos,offsets=offsets))

  end_time = time.time()
  execution_time = end_time - start_time
  if display:
    print(f"simulate time:{execution_time} s")


  savemat(output_path,{'data':signal})
  if display:
    fig,ax = plt.subplots(1,1)

    plot_radar_data(ax,signal,'simulate data')

    plt.show()
if __name__ == '__main__':
  simulate('/home/zhang_muxin/shapenetS2M_HPR/笔记本电脑/1/models/0.obj','/tmp/temp.mat')