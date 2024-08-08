#!/user/bin/env bash
################################################################################
# @Copyright: 2019-2023 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2023-08-22 10:47:20
# @Details  :
################################################################################

set -euxo pipefail

if [ $# != 2 ]; then
 echo "Usage: $0 data_csv_path device_id"
 exit
fi

test_data=$1
device_id=$2
http_server_address="http://0.0.0.0:80/generate"
# http_server_address="http://0.0.0.0:8000/v1/chat/interactive"

function run()
{
  local bs=$1
  local out_len=$2

  python3 profile_restful_api.py \
    ${http_server_address} \
    ${test_data} \
     --concurrency ${bs} \
     --out_len ${out_len} \
     --samples 200 \
     --top_k 3 \
     --top_p 0.95 \
     --temperature 0.0 \
     --repetition_penalty 1.15 \
     --device_id ${device_id} \
     --log_path 'perf.log'
}

run 16 256
exit 0

for out_len in {256,512,1024}
do
  for bs in {1,2,4,8,16,32,64,96,128,160,256}
    do
      echo ${bs} ${out_len}
      run ${bs} ${out_len}
    done
done
