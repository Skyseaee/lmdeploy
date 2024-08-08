#!/bin/bash
# 设置 PYTHONPATH
export PYTHONPATH=$(pwd)

# 读取模型名称作为变量
MODEL_NAME=$1

# 设置其他变量
MODEL_DIR="../models/${MODEL_NAME}/"
DEST_PATH="./models/lmdeploy-${MODEL_NAME}-fp8-hf"
COMPRESSED_MODEL="${MODEL_NAME}-fp8-hf"

TP=$2
EP=$3

# 进入脚本目录并转换模型
cd scripts
bash convert_compress_model.sh "${MODEL_DIR}" hf fp8 0

# 移动转换后的模型
mv "${COMPRESSED_MODEL}" "../models/"

# 返回上一级目录
cd ..

# 转换为 lmdeploy 格式
python -m lmdeploy convert llama2 "./models/${COMPRESSED_MODEL}/" --dst-path "${DEST_PATH}" --model-format fp8 --tp "${TP}" "${EP}" 