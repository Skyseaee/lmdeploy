#!/user/bin/env bash
################################################################################
# @Copyright: 2019-2023 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2023-08-22 10:47:20
# @Details  :
################################################################################

set -euxo pipefail

if [ $# != 4 ]; then
  echo "Usage: $0 workspace_path instance_num data_csv_path device_id(0 or 0,1 or 2,3 or 0,1,2,3)"
 exit
fi

workspace_path=$1
instance_num=$2
test_data=$3
device_id=$4

gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
monitor_device_id=$(echo "$device_id" | awk -F',' '{print $NF}')

service_name=0.0.0.0
service_port=8000
http_server_address="http://0.0.0.0:${service_port}/generate"
# http_server_address="http://0.0.0.0:${service_port}/v1/chat/interactive"

IMAGE_TAG='harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-lmdeploy:83cda14d'
CONTAINER_NAME=lmdeploy-test

function start_service(){
  local tp=$1

  # start service
  docker run -d --gpus all --env CUDA_VISIBLE_DEVICES=$device_id --privileged --shm-size=5g \
    -v ${PWD}:/workspace  \
    -p ${service_port}:${service_port} \
    --rm --name=${CONTAINER_NAME} ${IMAGE_TAG} \
    lmdeploy serve api_server ${workspace_path} \
    --server_name ${service_name} \
    --server_port ${service_port} \
    --instance_num ${instance_num} \
    --tp ${tp}

  # wait 1 minute for service ready
  sleep 1m
}


function stop_service(){
  docker stop ${CONTAINER_NAME}
}


function profile()
{
  local bs=$1
  local out_len=$2

  python3 profile_restful_api.py \
    ${http_server_address} \
    ${test_data} \
     --concurrency ${bs} \
     --out_len ${out_len} \
     --samples 200 \
     --top_k 3 \
     --top_p 0.95 \
     --temperature 0.0 \
     --repetition_penalty 1.15 \
     --device_id ${monitor_device_id} \
     --log_path 'perf.log'
}


for tp in 1 2 4;
do
  start_service $tp

  for out_len in 256 512 1024;
  do
    for bs in 1 2 4 8 16;
    do
      profile $bs ${out_len}
    done
  done

  stop_service
done
