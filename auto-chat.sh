#!/bin/bash

# 默认量化类型为 Q4
quant="Q4_K_M"
threads=6

# 参数解析
while [[ "$1" =~ ^- ]]; do
    case "$1" in
        -3) quant="Q3_K_M" ;;
        -4) quant="Q4_K_M" ;;
        -5) quant="Q5_K_M" ;;
        -8) quant="Q8_0" ;;
        -t) shift; threads="$1" ;;
        *) break ;;  # 传给 llama-cli 的其他参数
    esac
    shift
done

# 构建模型路径
model_path="./models/llama-2-7b/llama-2-7b-chat.${quant}.gguf"

# 输出配置信息
echo "Launching chat with:"
echo "Model: $model_path"
echo "Threads: $threads"
echo "Extra args: $@"

# 启动 llama-cli
./build/bin/llama-cli -m "$model_path" --chat-template llama2 -t "$threads" "$@"
