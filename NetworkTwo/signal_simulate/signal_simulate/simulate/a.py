from main import frame_simulate
from signal_simulate.util.mesh import TriMesh
import numpy as np

mesh = TriMesh.load_from_obj('/home/muxin/hdd/signal_simulate/data/mesh/display_after_HPR_decimate.obj')

radar_pos = np.array([0,0.61,0.08])

a =  frame_simulate(mesh=mesh,radar_pos=radar_pos)

print(a)

