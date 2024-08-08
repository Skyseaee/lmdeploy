#! /usr/bin bash

set -euxo pipefail

IMAGE_TAG=helm:test-quant

docker run -it --gpus all --privileged \
            -v ${PWD}:/my_scripts \
            -p 8098:8098 \
            --rm --name=helm-test ${IMAGE_TAG} /bin/bash
