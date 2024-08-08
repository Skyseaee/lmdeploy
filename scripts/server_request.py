import csv
import os
import json
import time
import requests
import argparse

parser = argparse.ArgumentParser(description='AIP LLM client params arg parser')
parser.add_argument('-i',
                    '--instance_id',
                    type=int,
                    default=0,
                    help='The number of instance_id.')
parser.add_argument('-s',
                    '--save_path',
                    type=str,
                    required=False,
                    default="",
                    help='path to output json file, not save to file is not set')
parser.add_argument('--stream', action='store_true',
                    help='Enable stream mode, default False')
args = parser.parse_args()

os.environ["PYTHONIOENCODING"] = "utf-8"

url = "http://0.0.0.0:80/generate"
# url = "http://sg9.aip.mlp.shopee.io/aip-svc-11/llm-lmdeploy-merge-yk/generate"

# get test_data
with open('data.csv', 'r') as fp:
    reader = csv.DictReader(fp)
    test_data = {}
    for row in reader:
        for column, value in row.items():
            test_data.setdefault(column, []).append(value)
    test_prompt = test_data['prompt']
    # test_anser = test_data['answer']

# set params
batch_size = 1
output_len = 1024

top_k = 3
top_p = 0.95
beam_width = 1
temperature = 0
repetition_penalty = 1.15


instance_id = 0  # threading.get_native_id() % 2  # 2 instance in service
if args.instance_id:
    instance_id = args.instance_id
print(f"instance_id: {args.instance_id}")


def get_streaming_response(response):
    for chunk in response.iter_lines():
        if chunk == b"\n":
            continue
        if chunk:
            payload = chunk.decode('utf-8')
            if payload.startswith("data:"):
                data = json.loads(payload.lstrip("data:").rstrip("/n"))
                traceid = data.pop('traceid', '')
                output = data.pop('text', '')
                input_tokens = data.pop('input_tokens', 0)
                generated_tokens = data.pop('generated_tokens', 0)
                history_tokens = data.pop('history_tokens')
                cost_time = data.pop('cost_time', '0.0')
                finish_reason = data.pop('finish_reason', None)
                finished = data.pop('finished', None)
                yield traceid, output, input_tokens, generated_tokens, cost_time, finish_reason, finished
            else:
                print(payload)


stream = args.stream

if args.save_path:
    with open(args.save_path, 'a') as json_file:
        json_file.write("[{}\n")  # save as list of dict

# send request
for i in range(0, len(test_prompt)):
    prompt = test_prompt[i]
    # base_answer = test_anser[i]
    if not args.save_path:
        user_input = input('Press any to continue...')
        if user_input:
            prompt = user_input
    print(f'Case {i} prompt len: {len(prompt)}\n')
#    print(f'Base: {base_answer}\n\n')

    headers = {'User-Agent': 'Test Client'}
    data = {
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,

        'instance_id': instance_id,
        'question': prompt,
        'output_len': output_len,
        'stream': stream,
        'random_seed': 42,
    }
    input_dict = {
        'traceid': 123456,
        'data': data
    }
    req = json.dumps(input_dict)
    start = time.time()
    res = requests.post(url=url, headers=headers, data=req, stream=stream)
    token_num = 0
    first_token_time = 0
    if stream:
        for traceid, output, input_tokens, generated_tokens, cost_time, finish_reason, finished in get_streaming_response(res):
            if token_num == 0:
                first_token_time = float(cost_time)
            if finished == True:
                total_time = float(cost_time)
            token_num = generated_tokens
            if finish_reason == 'length':
                print('WARNING: exceed session max length')
                continue
            print(output, end='', flush=True)

        total_time = time.time() - start
        print(
            f'\ntoken_num: {token_num}, first_token_time: {first_token_time:.2f}, totle_time: {total_time:.2f}\n')
    else:
        if res.status_code != 200:
            print(f'Error from LLM server {res.json()}')
        else:
            answer = res.json().get("text", "")
            print(res.json())
            # print("answer:", answer)
            res_json = res.json()
            res_json['prompt'] = prompt
            if args.save_path:
                with open(args.save_path, 'a') as json_file:
                    json_file.write(',\n' + json.dumps(res_json))

if args.save_path:
    with open(args.save_path, 'a') as json_file:
        json_file.write("]\n")  # save as list of dict
