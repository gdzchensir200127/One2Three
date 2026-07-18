from signal_simulate.util.mesh import TriMesh
from signal_simulate.util.plot import plot_radar_data,plot_iq_data
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
import copy
from scipy.io import savemat
import time
from multiprocessing import Process, Queue, Lock
import random,sys
from scipy.signal import gausspulse

warnings.filterwarnings('error', category=np.ComplexWarning)
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

bins = 192

# trans signal
# tao = (2*math.pi*fb*(math.log10(math.e))**(1/2))**(-1)

# g = lambda t:np.exp(-(np.square(t))/(2*(tao**2)))

g = lambda t:gausspulse(t,fc=fc,bw=bw,retquad=True,retenv=True)[2]

cos_carrier = lambda t:np.cos(2*np.pi*fc*t)
sin_carrier = lambda t:np.sin(2*np.pi*fc*t)

# sample time

t = np.arange(0,bins*sample_interval,sample_interval)

sample_point = t.size

def frame_simulate(mesh:TriMesh, radar_pos:np.ndarray) -> np.ndarray:
  '''
  simulate the one frame of the UWB radar signal reflected by the object
  param mesh: the TriMesh of the object
  param radar_pos: the position numpy array of the radar [x,y,z]
  '''
  face_num = mesh.faces.shape[0]

  # [face_num * t] array of recieve signal reflected by each face
  receive_signal = np.zeros((face_num,sample_point),dtype=np.complex128)
  
  for i in range(face_num):
    face = mesh.faces[i].tolist()
    vert1 = mesh.verts[face[0]-1]
    vert2 = mesh.verts[face[1]-1]
    vert3 = mesh.verts[face[2]-1]
    vec1 = vert2-vert1
    vec2 = vert3-vert1
    normal_vec = np.cross(vec1, vec2)
    if np.linalg.norm(normal_vec) == 0:
      continue
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    midpoint = (vert1+vert2+vert3)/3
    signal_vec = midpoint - radar_pos
    signal_vec = signal_vec / np.linalg.norm(signal_vec)
    dot_product = np.dot(normal_vec, signal_vec)
    norm_A = np.linalg.norm(normal_vec)
    norm_B = np.linalg.norm(signal_vec)

    angle = np.arccos(dot_product / (norm_A * norm_B))
    theta = (angle if angle < math.pi/2 else math.pi - angle)*2

    R = np.linalg.norm(midpoint-radar_pos)  # distance
    Tp = 2*R/light_speed  # TOF
    # Q component
    Q = (1/2) * g(t-Tp) * cos_carrier(Tp)
    # I component
    I = (1/2) * g(t-Tp) * sin_carrier(Tp)
    # signal strength
    sigma_square = 40
    orientation_factor = math.exp(-(theta**2)/(2*sigma_square))
    area_factor =  0.5 * np.linalg.norm(np.cross(vec1, vec2))
    distance_factor = 1/(R**2)
    strength = orientation_factor*area_factor*distance_factor
    signal = (I+Q*1j)*strength

    receive_signal[i,:] = signal

  # get the sum signal of each face

  receive_signal = np.sum(receive_signal,axis=0)

  return receive_signal

def object_move_signal_simulate(init_mesh:TriMesh,radar_pos:tuple,offsets:np.ndarray) -> np.ndarray:
  '''
  simulate the signal of a move object
  : param init_mesh: the init TiiMesh of the object
  : param radar_pos: the position numpy array of the radar [x,y,z]
  : param offsets: the position offset numpy array of the moving object, a [N * 3] array, N is the moving steps,
  each row is the position offset of each move step, such as [[x_offset1,y_offset1,z_offset1],...] 
  '''
  steps = offsets.shape[0]
  signal = np.zeros((steps,sample_point),dtype=np.complex128)
  for i in range(steps):
    offset = offsets[i]
    mesh = copy.deepcopy(init_mesh)
    mesh.verts += offset
    signal[i:,] = frame_simulate(mesh=mesh,radar_pos=radar_pos)
    pass

  return signal

def parallel_simulate(init_mesh:TriMesh,radar_pos:tuple,offsets:np.ndarray,result_queue:Queue,process_no:int,lock:Lock):
  '''
  simulate the partial signal and put into the result queue
  '''
  partial_signal = object_move_signal_simulate(init_mesh=init_mesh,radar_pos=radar_pos,offsets=offsets)
  lock.acquire()
  result_queue.put((process_no,partial_signal,))
  lock.release()

def merge_signal(process_num,result_queue:Queue,lock:Lock):
  start_time = time.time()
  results = []
  while True:
    lock.acquire()
    if not result_queue.empty():
      results.append(result_queue.get())
      lock.release()
    else:
      lock.release()
      time.sleep(0.1)

    if(len(results)>=process_num):
      break
  results.sort(key=lambda x:x[0])  # sort the results based on the process no

  signal = np.vstack([result[1] for result in results])
  
  end_time = time.time()
  execution_time = end_time - start_time
  print(f"simulate time:{execution_time} s")


  savemat('/home/muxin/hdd/signal_simulate/data/simulate_data/display_60.mat',{'data':signal})

  fig,ax = plt.subplots(1,1)

  plot_radar_data(ax,signal,'simulate data')

  plt.show()


if __name__ == '__main__':

  # mesh = TriMesh.load_from_obj('/home/muxin/hdd/signal_simulate/data/mesh/cylinder.obj')
  mesh = TriMesh.load_from_obj('/home/muxin/hdd/signal_simulate/data/mesh/display_after_HPR_decimate.obj')

  radar_pos = np.array([0,0.60,0.08])


  FPS = 40   # 40Hz FPS
  moving_speed = 0.01 # 1cm/s
  moving_step = moving_speed/FPS
  x_start = -0.35
  x_end = 0.35
  steps = int((x_end - x_start) / moving_step) + 1
  x_offsets = np.linspace(x_start,x_end,steps)
  offsets = np.zeros((steps,3),dtype=float)
  offsets[:,0:1] = x_offsets.reshape(-1,1)

  # y_start = 0
  # y_end = -0.1
  # steps = int(np.abs(y_end - y_start) / moving_step) + 1
  # y_offsets = np.linspace(y_start,y_end,steps)
  # offsets = np.zeros((steps,3),dtype=float)
  # offsets[:,1:2] = y_offsets.reshape(-1,1)



  result_queue = Queue()
  lock = Lock()

  process_num = 15 # 8 core cpu
  steps_per_process = math.ceil(steps/process_num)

  merge_process = Process(target=merge_signal,args=(process_num,result_queue,lock,))

  merge_process.start()

  processes = []
  for i in range(process_num):
    partial_offsets = offsets[i*steps_per_process:(i+1)*steps_per_process,:]
    processes.append(Process(target=parallel_simulate,args=(mesh,
                                                            radar_pos,
                                                            partial_offsets,
                                                            result_queue,
                                                            i,
                                                            lock,
                                                            )))
    
  for process in processes:
    process.start()
  for process in processes:
    process.join()

  merge_process.join()

  print(f'all processes waited')






