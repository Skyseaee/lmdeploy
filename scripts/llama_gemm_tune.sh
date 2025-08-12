#!/bin/bash

set -euxo pipefail

lmdeplot_install_path=$(python3 -c "import lmdeploy as _; print(_.__path__[0])")
llama_gemm=${lmdeplot_install_path}/bin/llama_gemm

for bs in {1..4}; do
    for tp in {1,2,4}; do
        echo "tuning bs: $bs with tp: $tp"
        $llama_gemm $bs \
                 1 \
                 512 \
                 40 \
                 128 \
                 13824 \
                 32000 \
                 1 \
                 $tp \
                 1
    done
done
