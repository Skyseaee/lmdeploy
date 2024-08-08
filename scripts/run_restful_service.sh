#!/usr/bin/env bash

set -euxo pipefail

if [ $# != 3 ]; then
  echo "Usage: $0 instance_num workspace_path device_id(0 or 0,1 or 1,3)"
  exit
fi

instance_num=$1
workspace=$2
device_id=$3
gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

service_name=0.0.0.0
service_port=8000

# https://github.com/InternLM/lmdeploy/blob/main/docs/en/restful_api.md

# CUDA_VISIBLE_DEVICES=$device_id nsys profile -o profile_4A30_gen256_bs64 lmdeploy serve api_server \
CUDA_VISIBLE_DEVICES=$device_id lmdeploy serve api_server \
  ${workspace} \
  --server_name ${service_name} \
  --server_port ${service_port} \
  --instance_num ${instance_num} \
  --tp ${gpu_num} \
  --log_level INFO
