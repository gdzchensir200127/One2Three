import torch
from emd import earth_mover_distance

# 设置随机种子以确保结果可复现
# torch.manual_seed(46)

# 定义批次大小和点的数量
B = 1  # 批次大小
N1 = 100  # p1 的点数量
N2 = 150  # p2 的点数量

# 创建两个点云张量（随机生成）
p1 = torch.rand(B, N1, 3).cuda()  # B x N1 x 3
p2 = torch.rand(B, N2, 3).cuda()  # B x N2 x 3

# 调用 EMD 计算函数
d = earth_mover_distance(p1, p2, transpose=False)

# 输出结果
print("EMD for each batch:", d)