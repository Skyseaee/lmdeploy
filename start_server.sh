#!/bin/bash
set -exo pipefail

model_dir=${AIP_MODEL_PATH:-models}
config_file=$(find ${model_dir} -name "config.yaml")
if [[ -z "$config_file" ]]; then
  echo "No config.yaml file exist, try to load with default parameters!"
  config_file=$(find ${model_dir} -name "config.json")
fi
echo "Running service using config version $config_file...\n"

# mass log storage
if [[ -z "${POD_NAME}" ]]; then
  echo "Environment variable POD_NAME not set"
else
  ln -s /root/log/$POD_NAME log
fi

if [[ $AIP_USER_PROTOCOLS == http* ]]; then
  echo "launching http service\n"
  python3 scripts/server_http.py $config_file
elif [[ $AIP_USER_PROTOCOLS == grpc* ]]; then
  echo "launching grpc service\n"
  echo "llm grpc service not supported yet!\n"
else
  echo "not supported protocal: $AIP_USER_PROTOCOLS\n"
fi
