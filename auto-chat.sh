#!/bin/bash

# 默认量化类型为 Q4
quant="Q4_K_M"

# 检查是否有 -3、-4、-5 等参数来覆盖默认
while [[ "$1" =~ ^- ]]; do
    case "$1" in
        -3) quant="Q3_K_M" ;;
        -4) quant="Q4_K_M" ;;
        -5) quant="Q5_K_M" ;;
        -8) quant="Q8_0" ;;
        *) echo "Unknown option: $1" && exit 1 ;;
    esac
    shift
done

# 构建模型路径
model_path="./models/llama-2-7b/llama-2-7b-chat.${quant}.gguf"

# 启动 llama-cli
./build/bin/llama-cli -m "$model_path" --chat-template llama2 "$@" -t 6
