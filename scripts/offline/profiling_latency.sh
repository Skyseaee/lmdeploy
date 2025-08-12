#!/user/bin/env bash
################################################################################
# @Copyright: 2024 Shopee. All Rights Reserved.
# @Author   : meng.liu@shopee.com
# @Date     : 2024-07-30
# @Details  : analysis script for LLMs latency.
################################################################################

DEVICE="H100"
FRAMEWORK_TYPE="LMDeploy"
#FRAMEWORK_TYPE="TRT-LLM"
MODEL_TYPE="Compass-13B"

trt_engine_path="/workspace/trtllm-v0.13.0-meng/scripts/profiling/trt_engines/Llama-2-13b-hf_fp8"
tokenizer_dir="/workspace/trtllm-v0.13.0-meng/models/Llama-2-13b-hf"

if [ $# -gt 0 ]; then
    DEVICE=$1
fi

if [ $# -gt 1 ]; then
    FRAMEWORK_TYPE=$2
fi

if [ $# -gt 2 ]; then
    MODEL_TYPE=$3
fi

echo " *** 开始${MODEL_TYPE} on ${FRAMEWORK_TYPE} ${DEVICE} 的性能测试"

logdir="./infer_latency/${DEVICE}_${FRAMEWORK_TYPE}_${MODEL_TYPE}/log"
ns_logdir="./infer_latency/${DEVICE}_${FRAMEWORK_TYPE}_${MODEL_TYPE}/ns"

if [ ! -f ${logdir} ] ; then
    mkdir ${logdir} -p
fi

if [ ! -f ${ns_logdir} ] ; then
    mkdir ${ns_logdir} -p
fi

all_log="${logdir}/all-log.log"


# BatchSize:[1-16]
for BatchSize in 1 2 4 8 16 32 64 128 256;
do

    TP=1
    InputLen=673
    OutputLen=270

    tmp_log_ths=${logdir}/batchsize-${BatchSize}-inputlen-${InputLen}-outputlen-${OutputLen}.log
    nsysprofout_ths=${ns_logdir}/batchsize-${BatchSize}-inputlen-${InputLen}-outputlen-${OutputLen}.nsysprofout

    nvidia-smi >> ${tmp_log_ths}

    #echo "Star nsys profile -o ${nsysprofout_ths} --stats=true --force-overwrite true ./bin/llama_triton_example ../examples/cpp/llama/llama_config.ini \
    # ../examples/cpp/llama/start_ids.csv ${BatchSize} >> ${tmp_log_ths}"
    
    if [ $FRAMEWORK_TYPE == "LMDeploy" ] ; then
        nsys profile -t cuda,nvtx -o ${nsysprofout_ths} --stats=true --force-overwrite true ./bin/llama_triton_example \
            ../examples/cpp/llama/llama_config.ini \
            ../examples/cpp/llama/start_ids.csv ${BatchSize} >> ${tmp_log_ths}
    elif [ $FRAMEWORK_TYPE == "TRT-LLM" ] ; then
        nsys profile -t cuda,nvtx -o ${nsysprofout_ths} --stats=true --force-overwrite true \
                mpirun --allow-run-as-root -n ${TP} python3 ../examples/run.py \
                --engine_dir ${trt_engine_path} \
                --tokenizer_dir ${tokenizer_dir} \
                --max_output_len ${OutputLen} --max_input_length ${InputLen} \
                --batch_size ${BatchSize} --run_profiling >> ${tmp_log_ths}
    else
        echo "Un-Support Framework ${FRAMEWORK_TYPE}"
    fi

done #BatchSize
