# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import gausspulse
# import math

# # 时间范围
# t = np.linspace(-2e-9, 2e-9, 1000)

# # 中心频率和分数带宽
# fb = 1.4e9
# fc = 7.29e9
# bw = fb/fc  # 分数带宽

# # 生成高斯脉冲的包络
# envelope = gausspulse(t, fc=fc, bw=bw, retquad=True, retenv=True)[2]

# g = lambda t:gausspulse(t,fc=fc,bw=bw,retquad=True,retenv=True)[2]

# cos_carrier = lambda t:np.cos(2*np.pi*fc*t)
# sin_carrier = lambda t:np.sin(2*np.pi*fc*t)

# y = g(t)*cos_carrier(t)

# # 绘制包络波形
# plt.plot(t, y)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Gaussian Pulse Envelope')
# plt.grid(True)
# plt.show()

import numpy as np
import cupy as cp
import time
### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,100))
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,100))
e = time.time()
print(e - s)
### Numpy and CPU
s = time.time()
x_cpu *= 5
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu *= 5
e = time.time()
print(e - s)