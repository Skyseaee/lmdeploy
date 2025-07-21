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

if [ "$(pip list | grep setuptools-scm | wc -l)" -eq "0"  ]; then
  python3 -m pip install setuptools-scm
fi

VERSION=$(python3 -c "from setuptools_scm import get_version; print(get_version(version_scheme='no-guess-dev'))")
VERSION=$(echo "$VERSION" | sed 's/\.d[0-9]\{8\}//' | sed 's/+\(g\)\?/-/')

IMAGE_TAG=harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-lmdeploy:${VERSION}

if [ $1 = src ]; then
  docker build  -t ${IMAGE_TAG} -f docker/Dockerfile_cu125.aip .

elif [ $1 = pip ]; then
  docker build -t ${IMAGE_TAG} -f docker/Dockerfile .

else
  echo "Installation type only supports src or pip"
  exit
fi

# docker push $IMAGE_TAG
