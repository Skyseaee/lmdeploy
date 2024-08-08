#!/bin/sh

# https://git.garena.com/shopee/MLP/aip/toolchains/evaluation/helm
# docker pull harbor.shopeemobile.com/aip/helm:test

model_name=${1}
ctalk_token=${2}

run_specs_lists="${model_name}.conf"
run_specs_array=($run_specs_lists)

# export AIP_MODEL_URL="http://sg9.aip.mlp.shopee.io/services/82984/generate"
# export AIP_MODEL_URL="http://10.55.17.10:8099/generate"

for run_specs in "${run_specs_array[@]}"
do
    helm-run --conf-paths ${run_specs} --models-to-run shopee/searchGPT --suite ${model_name} -n 1 --max-eval-instances 1000 --local-path cache/${model_name}
done

helm-summarize --suite ${model_name}

# helm-parse --suite mmlu --ctalk-token gnsYG-aIQX6InC2pwtfynw
helm-parse --suite ${model_name} --ctalk-token ${ctalk_token}
