#!/bin/bash

# 默认量化类型为 Q4
quant="Q4_K_M"

# 默认线程组设置
threads="6"

# 默认并发并行度
prompts=16
n_tokens=16

# 默认轮次
rounds=1

# 参数解析（支持 -3/-4/-5/-8 和 --round）
while [[ "$1" =~ ^- ]]; do
    case "$1" in
        -0) quant="Q4_0" ;;
        -3) quant="Q3_K_M" ;;
        -4) quant="Q4_K_M" ;;
        -5) quant="Q5_K_M" ;;
        -8) quant="Q8_0" ;;
        -p) shift; prompts="$1" ;;      # 设置 prompt 参数
        -n) shift; n_tokens="$1" ;;    # 设置 token 数
        -t) shift; threads="$1" ;;     # 设置线程数列表
        -r) shift; rounds="$1" ;; # 设置轮次
        *) echo "Unknown option: $1" && exit 1 ;;
    esac
    shift
done

# 构建模型路径
model_path="./models/llama-2-7b/llama-2-7b-chat.${quant}.gguf"

# 输出配置
echo "Benchmarking model: $model_path"
echo "Threads: $threads"
echo "Prompts: $prompts"
echo "Tokens: $n_tokens"
echo "Rounds: $rounds"

# 循环执行指定轮次的 llama-bench
for ((i=1; i<=rounds; i++)); do
    echo "Running round $i of $rounds"
    ./build/bin/llama-bench \
      -m "$model_path" \
      -p "$prompts" \
      -n "$n_tokens" \
      -t "$threads" \
      --csv \
      --round "$i"
done