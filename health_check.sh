#!/bin/bash
set -exo pipefail

if [[ $AIP_USER_PROTOCOLS == http* ]]; then
  echo "health check http service...\n"
  curl --fail http://0.0.0.0:80/v1/models
elif [[ $AIP_USER_PROTOCOLS == grpc* ]]; then
  echo "health check grpc service...\n"
  echo "llm grpc service not supported yet!\n"
  exit 1
else
  echo "not supported protocal: $AIP_USER_PROTOCOLS"
  exit 1
fi
