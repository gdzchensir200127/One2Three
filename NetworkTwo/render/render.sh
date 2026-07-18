#!/bin/bash

# 定义参数列表
# params=("02933112" "03636649" "04090263" "04379243" "04530566" "02828884" "03211117" "03691459" "04256520" "04401088" "02691156")
params=("02691156")

# 遍历参数列表并调用 Python 脚本
for param in "${params[@]}"; do
    blender blank.blend -b -P render.py -- /home/zhang_muxin/shapenetS2M_init "$param" ./obj_list/"$param".list 128
done