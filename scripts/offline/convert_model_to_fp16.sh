#!/bin/bash

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd)


# 读取模型名称作为变量
MODEL_NAME=$1
TP=$2
EP=$3

# 设置其他变量
MODEL_DIR="../models/${MODEL_NAME}/"
DEST_PATH="./models/lmdeploy-${MODEL_NAME}-fp16"


# 转换为 lmdeploy 格式
python -m lmdeploy convert llama2 "./models/${MODEL_DIR}/" --dst-path "${DEST_PATH}" --tp "${TP}" "${EP}"