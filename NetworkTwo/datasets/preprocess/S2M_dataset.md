# mesh
两种文件
.obj 方便进行signal simulate
.dat 直接存储采样后点云坐标位置，方便读取GT

处理方法，读取shapenet原始的obj文件，首先进行合适的放缩等变换，储存为.obj，之后采样得到点云数据存储为.dat文件

需要考虑的事情：
放缩的尺寸，mesh的初始坐标

# signal
.dat or .mat 存储模拟的信号