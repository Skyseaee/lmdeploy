#!/user/bin/env bash
################################################################################
# @Copyright: 2019-2023 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2023-09-25 16:19:07
# @Details  :
################################################################################

set -euxo pipefail


if [ $# != 4 ]; then
 echo "Usage: $0 model_path precision(fp16,int4 or kv) num_tp device_id"
 exit
fi

MODEL_PATH=$1
precision=$2
num_tp=$3
dev_id=$4

export CUDA_VISIBLE_DEVICES=${dev_id}


quant_model_path=${MODEL_PATH}-int4
kv_cache_path=${MODEL_PATH}-kv-int8
kv_scales_path=${kv_cache_path}/kv_scales_${num_tp}

final_model_path=workspace_${precision}_${num_tp}

if [ $precision = fp16  ]; then
  # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/turbomind/deploy/converter.py#L139
  lmdeploy convert \
    --model_name llama \
    --model_path ${MODEL_PATH} \
    --model_format hf \
    --dst_path $final_model_path \
    --tp ${num_tp}

elif [ $precision = int4  ]; then
  if [ ! -d $quant_model_path ]; then
      mkdir -p $quant_model_path
  fi

  # Step1: Generate Quantization Parameter
  lmdeploy lite calibrate \
    --model ${MODEL_PATH} \
    --calib_dataset 'wikitext2' \
    --calib_samples 128 \
    --calib_seqlen 2048 \
    --work_dir $quant_model_path

  # Step2: Quantize Weights
  lmdeploy lite auto_awq \
    --model ${MODEL_PATH} \
    --w_bits 4 \
    --w_group_size 128 \
    --work_dir ${quant_model_path}

  # Step3: Convert model
  lmdeploy convert \
    --model-name llama \
    --model-path ${quant_model_path} \
    --model-format awq \
    --group-size 128 \
    --dst_path $final_model_path \
    --tp ${num_tp}

elif [ $precision = kv  ]; then
  if [ ! -d $kv_scales_path ]; then
      mkdir -p $kv_scales_path
  fi

  # Step1: Convert the Hugging Face model format to the TurboMind inference format to create a workspace directory
  lmdeploy convert \
    --model_name llama \
    --model_path ${MODEL_PATH} \
    --model_format hf \
    --dst_path $final_model_path \
    --tp ${num_tp}

  # Step2: Get the quantization parameters
    # get minmax
  lmdeploy lite calibrate \
    --model ${MODEL_PATH} \
    --calib_dataset 'wikitext2' \
    --calib_samples 128 \
    --calib_seqlen 2048 \
    --work_dir ${kv_cache_path}

    # get quant parameters
    # kv_qparams will generate fp32 scaling factors in the ${kv_scales_path} directory
  lmdeploy lite kv_qparams \
    --work_dir ${kv_cache_path} \
    --turbomind_dir ${kv_scales_path} \
    --kv_sym False \
    --num_tp ${num_tp}

  # Step3: Copy the scaling factors into workspace/triton_models/weights/
  cp -r ${kv_scales_path}/* $final_model_path/triton_models/weights/

  # Step4: Modify workspace/triton_models/weights/config.ini
    # Set use_context_fmha to 0, which means turning off flashattention
    # Set quant_policy to 4. This means enabling kv_cache int8

else
  echo "Precision only supports fp16, int4 or kv"
  exit
fi

#if you want to launch http/restful service, please run `bash run_http_service.sh`
#if you want to launch TritonGRPCService, please run `bash workspace/service_docker_up.sh`
