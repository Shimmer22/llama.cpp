#!/bin/bash

# 默认量化类型为 Q4
quant="Q4_K_M"

# 默认线程组设置（可以覆盖）
threads="4,6,8"

# 默认并发并行度
prompts=16
n_tokens=16

# 参数解析（支持 -3/-4/-5/-8）
while [[ "$1" =~ ^- ]]; do
    case "$1" in
        -3) quant="Q3_K_M" ;;
        -4) quant="Q4_K_M" ;;
        -5) quant="Q5_K_M" ;;
        -8) quant="Q8_0" ;;
        -p) shift; prompts="$1" ;;      # 设置 prompt 参数
        -n) shift; n_tokens="$1" ;;      # 设置 token 数
        -t) shift; threads="$1" ;;       # 设置线程数列表
        *) echo "Unknown option: $1" && exit 1 ;;
    esac
    shift
done

# 构建模型路径
model_path="./models/llama-2-7b/llama-2-7b-chat.${quant}.gguf"

# 输出配置
echo "Benchmarking model: $model_path"
echo "Threads: $threads"
echo "Prompts: $parallel"
echo "Tokens: $n_tokens"

# 启动 llama-bench
./build/bin/llama-bench \
  -m "$model_path" \
  -p "$parallel" \
  -n "$n_tokens" \
  -t "$threads"
