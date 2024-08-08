"""
Refer to: https://github.com/InternLM/lmdeploy/blob/main/benchmark/profile_restful_api.py
"""

import os
import json
import random
import time
import subprocess
import requests
from queue import Queue
from threading import Thread
import fire
import pandas as pd
import numpy as np
import logging


def get_gpu_memory(device_idx: int):
    command = f"nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv --id={device_idx}"
    gpu_mem_util = subprocess.check_output(command.split()).decode("ascii").split("\n")[1].split(',')
    gpu_mem_mb = int(gpu_mem_util[0].split()[0]) # MiB
    gpu_util = int(gpu_mem_util[1].split()[0]) # %
    return gpu_mem_mb, gpu_util


def get_streaming_response(prompt: str,
                           server_addr: str,
                           instance_id: int,
                           request_output_len: int,
                           top_k: int,
                           top_p: float,
                           temperature: float,
                           repetition_penalty: float,
                           stream: bool = True):
    headers = {'User-Agent': 'Test Client'}
    data = {
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,

        'instance_id': instance_id,
        'question': prompt,
        'output_len': request_output_len,
        'stream': stream,
        'random_seed': 42,
        'stop': False,
    }
    input_dict = {
        'traceid': 123456,
        'data': data
    }
    req = json.dumps(input_dict)
    res = requests.post(server_addr, headers=headers,
                        data=req, stream=stream)

    for chunk in res.iter_lines():
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
                yield output, input_tokens, generated_tokens


def infer(server_addr: str, session_id: int, out_len: int, top_k: int, top_p: float,
          temperature: float, repetition_penalty: float, req_queue: Queue,
          res_que: Queue):
    stats = []
    while not req_queue.empty():
        try:
            prompt = req_queue.get(timeout=5)
        except Queue.Empty as exc:
            logger.warn(exc)
            break

        timestamps = []
        tokens = []
        timestamps.append(time.perf_counter())
        for res, input_tokens, token in get_streaming_response(
                prompt,
                server_addr,
                session_id,
                request_output_len=out_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = timestamps[1] - timestamps[0]
        res_latency = timestamps[-1] - timestamps[0]
        out_token_len = tokens[-1] - tokens[0]

        logging.info('[profile_restful_api] - ' +
            f'request info: session {session_id}, ' +
            f'input_seqlen {input_tokens}, output_seqlen {out_token_len}, '
            f'latency: {int(res_latency*1000)} ms, '
            f'first_token_latency: {int(first_token_latency*1000)} ms')

        stats.append([first_token_latency, input_tokens, out_token_len, res_latency])
    res_que.put((session_id, stats))


def warmup(server_addr: str,
           concurrency: int,
           out_len: int,
           prompts_list: list,
           top_k: int = 3,
           top_p: float = 0.95,
           temperature: float = 0.0,
           repetition_penalty: float = 1.15,
           warmup_round: int = 1):
    logging.info('start to warmup ...')

    def _infer(server_addr, session_id):
        for _ in range(warmup_round):
            tokens = []
            for _, _, token in get_streaming_response(
                    prompt=prompts_list[0],
                    server_addr=server_addr,
                    instance_id=session_id,
                    request_output_len=out_len,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty):
                tokens.append(token)
                continue
            logging.info(f'warmup gen_len: {tokens[-1] - tokens[0]}')

    _start = time.perf_counter()
    procs = []
    for i in range(concurrency):
        proc = Thread(target=_infer, args=(server_addr, i + 1))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    logging.info(f'end warmup, elapsed time: {round(_end - _start, 2)} s')


def read_dataset(dataset_path: str, samples: int):
    start = time.perf_counter()
    df = pd.read_csv(dataset_path, index_col=None)
    prompts = list(df['prompt'])

    logging.info(f'elapsed time for read data: '
          f'{round(time.perf_counter() - start, 2)} s')

    filtered_dataset = []
    samples = min(samples, len(prompts))
    if samples > 0:
#        filtered_dataset = random.sample(prompts, samples)
        filtered_dataset = prompts[:samples]
    else:
        logging.error('samples number should > 0')

    que = Queue()
    for data in filtered_dataset:
        que.put(data)
    return que, len(filtered_dataset), filtered_dataset


def main(server_addr: str,
         dataset_path: str,
         concurrency: int = 1,
         out_len: int = 256,
         samples: int = 1,
         top_k: int = 3,
         top_p: float = 0.95,
         temperature: float = 0.0,
         repetition_penalty: float = 1.15,
         device_id: int = 0,
         log_path: str = 'perf.log'):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()
                                  ]
                        )

    req_queue, n_req, prompts_list = read_dataset(dataset_path, samples)
    warmup(server_addr, concurrency, out_len, prompts_list)

    res_que = Queue()
    procs = []

    _start = time.perf_counter()
    for i in range(concurrency):
        proc = Thread(target=infer,
                      args=(server_addr, i + 1, out_len, top_k, top_p, temperature,
                            repetition_penalty, req_queue, res_que))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    elapsed_time = _end - _start

    stats = []
    while not res_que.empty():
        session_id, _stats = res_que.get()
        logging.info(f'\n{"-" * 50}\n'
              f'session {session_id} stats: \n{_stats}\n{"-" * 50}\n')
        if not _stats:
            continue
        stats.extend(_stats)

    logging.info(f'Finished request number: {len(stats)} (Request number: {n_req})')
    assert len(stats) == n_req, f'the stats list len is {len(stats)} != {n_req}'
    stats = np.array(stats).reshape(-1, 4)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_avg = np.mean(stats[:, 0], axis=0)
    input_token_avg = np.sum(stats[:, 1], axis=0) / n_req
    token_generated = np.sum(stats[:, 2], axis=0)
    gen_token_avg = token_generated / n_req
    res_latency_list = stats[:, 3].squeeze()

    latency_p50 = np.percentile(res_latency_list, 50)
    latency_p90 = np.percentile(res_latency_list, 90)
    latency_p99 = np.percentile(res_latency_list, 99)
    latency_avg = np.mean(res_latency_list)
    token_throughput = token_generated / elapsed_time
    req_throughput = n_req / elapsed_time

    if "mlp.shopee.io" in server_addr:
        gpu_mem_mb, gpu_util = None, None
    else:
        gpu_mem_mb, gpu_util = get_gpu_memory(device_id)

    data = {"Gen_Token_Len Setting": out_len,
            "Batch Size/Concurrency": concurrency,
            "Avg_Input_Token_Len": round(input_token_avg, 2),
            "Avg_Gen_Token_Len": round(gen_token_avg, 2),
            "GPU_Mem_Usage/MiB": gpu_mem_mb,
            "GPU-Util/%": gpu_util,
            "Elapse_Time/s": round(elapsed_time, 3),
            "First_Token_Latency/s": round(first_token_latency_avg, 3),
            "Latency_P50/s": round(latency_p50, 3),
            "Latency_P90/s": round(latency_p90, 3),
            "Latency_P99/s": round(latency_p99, 3),
            "Latency_AVG/s": round(latency_avg, 3),
            "Token QPS(token/s)": round(token_throughput, 2),
            "QPS(req/s)": round(req_throughput, 2),
            }
    df = pd.DataFrame([data])
    df = df.transpose()
    df.columns = ["" for i in range(len(df.columns))]
    logging.info('Performance Summary' \
                 f'{df.to_markdown(tablefmt="simple", numalign="left", stralign="left")}\n'
                 )

    # output to csv
    log_name, _ = os.path.splitext(log_path)
    csv_file_path = log_name + ".csv"
    header = "Gen_Token_Len Setting, Batch Size/Concurrency, Avg_Input_Token_Len, Avg_Gen_Token_Len, " \
             "GPU_Mem_Usage/MiB, GPU-Util/%, Elapse_Time/s, First_Token_Latency/s, Latency_P50/s, " \
             "Latency_P90/s, Latency_P99/s, Latency_AVG/s, Token QPS(token/s), QPS(req/s)\n"
    with open(csv_file_path, 'a') as f:
        if not f.tell():
            f.write(header)
        line = f'{out_len},{concurrency},{input_token_avg:.2f},{gen_token_avg:.2f},' \
               f'{gpu_mem_mb},{gpu_util},{elapsed_time:.3f},{first_token_latency_avg:.3f},' \
               f'{latency_p50:.3f},{latency_p90:.3f},{latency_p99:.3f},{latency_avg:.3f},' \
               f'{token_throughput:.2f},{req_throughput:.2f}\n'
        f.write(line)
        logging.info(f'Perf data have been saved in {csv_file_path}')


if __name__ == '__main__':
    fire.Fire(main)
