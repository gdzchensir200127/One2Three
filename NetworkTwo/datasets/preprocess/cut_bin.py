import os
import scipy.io as sio
import numpy as np

# 目标根目录（根据需求修改）
root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/fornet/Simulate_mask"

# 遍历根目录下所有子目录和文件
for root, dirs, files in os.walk(root_dir):
    for file_name in files:
        # 只处理.mat文件
        if file_name.endswith(".mat"):
            file_path = os.path.join(root, file_name)
            try:
                # 读取mat文件（保留所有变量）
                mat_data = sio.loadmat(file_path, squeeze_me=False)  # squeeze_me=False避免自动降维
                
                # 检查是否存在名为'data'的变量，且形状为(2880, 96)
                if "data" in mat_data:
                    data_array = mat_data["data"]
                    if data_array.shape == (2880, 96):
                        # 裁剪第11-19列（索引11:20，左闭右开）
                        mat_data["data"] = data_array[:, 11:20]
                        
                        # 覆盖保存到原文件（保持MATLAB兼容格式）
                        sio.savemat(
                            file_path,
                            mat_data
                        )
                        print(f"✅ 成功处理：{file_path}")
                    else:
                        # 形状不匹配，跳过
                        print(f"⚠️  跳过：{file_path} - 'data'形状为{data_array.shape}（需2800*96）")
                else:
                    # 无'data'变量，跳过
                    print(f"⚠️  跳过：{file_path} - 未找到'data'变量")
            
            except Exception as e:
                # 处理异常（如文件损坏、权限问题等）
                print(f"❌ 处理失败：{file_path} - 错误信息：{str(e)}")

print("\n🎯 批量处理完成！")