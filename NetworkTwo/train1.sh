#!/bin/bash

# 定义要替换的词汇列表
replacements=("笔记本电脑" "水杯" "显示器" "扳手" "手机" "刀具" "键盘" "易拉罐")

# 循环执行命令
for tool in "${replacements[@]}"; do
  echo "========================================"
  echo "正在处理: $tool"
  echo "========================================"
  
  # 执行命令（核心逻辑）
  python3 ./entrypoint_train.py \
    --name realdataset \
    --options "./experiments/signal2pixel/signal2pixel_4views_encoder_shared_out4_realdataset_unseen_${tool}.yaml"

  # 可选：添加间隔时间（单位：秒）
  # sleep 10
done

echo "所有任务执行完成！"
