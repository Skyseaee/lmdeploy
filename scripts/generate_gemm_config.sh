#! /usr/bin/bash

set -euxo pipefail

lmdeplot_install_path=$(python3 -c "import lmdeploy as _; print(_.__path__[0])")
llama_gemm=${lmdeplot_install_path}/bin/llama_gemm

# https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#run-gpt
# <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size> <is_append>
# Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)

for bs in 1 2 4 8 16 32 64 128 256 512;
do
  for tp in 1;
  do
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

