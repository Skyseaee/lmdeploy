'''
locust -f locust_test.py --headless --host=http://sg9.aip.mlp.shopee.io/aip-svc-123/test-search-vicuna-4g -u 32 -r 2 -t 3m --output_len=512
'''
import csv
import json
import random
import os

from locust import task, HttpUser, events

os.environ["PYTHONIOENCODING"]="utf-8"

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--output_len", type=int, default=512, help="output len")


# get test_data
with open('data.csv', 'r') as fp:
    reader = csv.DictReader(fp)
    test_data = {}
    for row in reader:
        for column, value in row.items():
            test_data.setdefault(column, []).append(value)
    test_data = test_data['prompt']

top_k = 3
top_p = 0.95
beam_width = 1
temperature = 0
repetition_penalty = 1.15

stream = False
stats = {"total_output_len": 0, "req_num": 0}

def build_request(output_len):
    prompt = random.choice(test_data)
    data = {
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'question': prompt,
        'output_len': output_len,
        'stream': stream,
        'random_seed': 42
    }
    input_dict = {
        'traceid': 123456,
        'data': data
    }
    req = json.dumps(input_dict)
    return req

# print(build_request(512))

class MyUser(HttpUser):

    @task
    def process(self):
        req = build_request(self.environment.parsed_options.output_len)

        with self.client.post("/generate", data=req, stream=stream) as res:
            if res.status_code != 200:
                print("Didn't detect bad response, got: " + str(res.status_code))
            else:
                stats["total_output_len"] += res.json()['generated_tokens']
                stats["req_num"] += 1

@events.test_stop.add_listener
def results(environment, **kw):
    print("total_output_len:", stats["total_output_len"])
    print("req_num:", stats["req_num"])
