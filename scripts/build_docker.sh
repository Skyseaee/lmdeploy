#!/user/bin/env bash
################################################################################
# @Copyright: 2019-2023 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2023-09-25 14:49:32
# @Details  :
################################################################################

#!/bin/bash

set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 install_type(src or pip)"
  exit
fi

VERSION=$(grep '^__version__' ../lmdeploy/version.py | grep -o '=.*' | tr -d "= '")
IMAGE_TAG=harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-lmdeploy:${VERSION}

if [ $1 = src ]; then
  docker build  -t ${IMAGE_TAG} -f ../docker/Dockerfile.aip ..

elif [ $1 = pip ]; then
  docker build -t ${IMAGE_TAG} -f Dockerfile .

else
  echo "Installation type only supports src or pip"
  exit
fi

# docker push $IMAGE_TAG
