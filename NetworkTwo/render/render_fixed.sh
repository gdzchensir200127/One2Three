#!/bin/bash

# # 定义参数列表
params=("02933112" "03636649" "04090263" "04379243" "04530566" "02828884" "03211117" "03691459" "04256520" "04401088" "02691156" "2958343" "03001627")

# 遍历参数列表并调用 Python 脚本
for param in "${params[@]}"; do
    blender blank.blend -b -P render_fixed.py -- /home/zhang_muxin/data/shapenet_obj "$param" ./obj_list/"$param".list 128 4
done
# 定义参数列表
# params=("笔记本电脑" "水杯" "显示器" "扳手" "手机" "刀具" "键盘")
# params=("笔记本电脑")
# 遍历参数列表并调用 Python 脚本
# for param in "${params[@]}"; do
#     blender blank.blend -b -P render_fixed.py -- /home/muxin/hdd/data/obj_init "$param" ./obj_list/"$param".list 128 4
# done