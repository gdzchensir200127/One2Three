#!/bin/bash

# 定义要替换的词汇列表
replacements=("0.0001" "0.00001" "0.00005" "0.000001" "0.000005")

# 循环执行命令
for lr in "${replacements[@]}"; do
  
  # 执行命令（核心逻辑）
  python3 ./entrypoint_train.py \
    --name finetune \
    --options "./experiments/signal2pixel/signal2pixel_4views_encoder_shared_out4_realdataset_unseen_finetune_${lr}.yaml"
  # 可选：添加间隔时间（单位：秒）
  # sleep 10
done

echo "所有任务执行完成！"
