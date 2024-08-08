#!/usr/bin/env bash
################################################################################
# @Copyright: 2019-2023 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2023-09-25 14:49:56
# @Details  :
################################################################################

set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 port"
  exit
fi

port=$1

IMAGE_TAG=harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-lmdeploy:0.5.3-7d95683e

docker run -it --gpus all --privileged --shm-size=10g \
            --network=host --ipc=host \
            -v ${PWD}:/workspace \
            -v /data/public_models:/data \
            -p ${port}:${port} \
            --rm --name=ld-test-${port} ${IMAGE_TAG} /bin/bash
