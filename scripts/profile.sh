#!/bin/bash

set -euxo pipefail

function run()
{
  local http_server_address=''
  local concurrency=1
  local request_output_len=512
  local samples=1000
  local top_k=10
  local top_p=0.85
  local temperature=0.3
  local repetition_penalty=1.15
  local log_path=perf.log
  local device_id=0

  local test_data=$1
  local pressure_config=$2

  while read line;do
    eval "$line"
  done < $pressure_config

  if [ -z "$http_server_address" ]; then
    echo "service address is empty!!!"
    exit 1
  fi

  if [ -z "$test_data" ]; then
    echo "test_data not found!!!"
    exit 1
  fi

  IFS=',' read -ra  con_list <<< "$concurrency"
  for bs in "${con_list[@]}"; do
    echo "run stress testing with concurrency: "$bs
    python3 profile_restful_api.py \
      ${http_server_address} \
      ${test_data} \
      --concurrency ${bs} \
      --out_len ${request_output_len} \
      --samples  ${samples} \
      --top_k ${top_k} \
      --top_p ${top_p} \
      --temperature ${temperature} \
      --repetition_penalty ${repetition_penalty} \
      --device_id ${device_id} \
      --log_path ${log_path}
  done
}
run $1 $2
