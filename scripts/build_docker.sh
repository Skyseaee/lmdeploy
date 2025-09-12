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

IMAGE_TAG=harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-onellm:${VERSION}
CACHE_IMAGE_TAG=harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-onellm/cache-image

if [ $1 = src ]; then
  DOCKERFILE_PATH=docker/Dockerfile_cu125.aip
elif [ $1 = pip ]; then
  DOCKERFILE_PATH=docker/Dockerfile.aip
else
  echo "Installation type only supports src or pip"
  exit 1
fi

docker pull $CACHE_IMAGE_TAG || true

DOCKER_BUILDKIT=1 docker build --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg VERSION=${VERSION} \
  --cache-from ${CACHE_IMAGE_TAG} \
  -t ${IMAGE_TAG} \
  -t ${CACHE_IMAGE_TAG} \
  -f ${DOCKERFILE_PATH} .

# docker push $IMAGE_TAG
docker push $CACHE_IMAGE_TAG
