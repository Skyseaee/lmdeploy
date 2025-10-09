import gc
import re
import os
import sys
import time
import json
import pytest
import shlex
import subprocess
import requests
from load_server_testdata import check_gpu


MAX_SERVER_START_WAIT_S = 240  # wait for server to start for 60 seconds

class ServerRunner:
    def __init__(self, model_path, service_args):
        self.proc = subprocess.Popen(
            ["lmdeploy", "serve", "api_server", model_path] + shlex.split(service_args),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                time.sleep(10)
                if requests.get(f"http://0.0.0.0:23333/v1/models").status_code == 200:
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                if time.time() - start > MAX_SERVER_START_WAIT_S:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    def terminate_and_wait(self):
        if not self.proc or self.proc.poll() is not None:
            return  # 进程已结束
        print(f"Terminiting ServerRunner...")
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()  # 确保进程已终止
        finally:
            self.proc = None

    def __del__(self):
        self.terminate_and_wait()


def quantize_model(model_path, precision, service_args):
    find_tp = re.search(r"--tp\s+(\d+)", service_args)
    device_id = ",".join(map(str, range(int(find_tp.group(1) if find_tp else "1"))))
    quant_model_path = f"{model_path}-{precision}-hf"
    config_file = os.path.join(quant_model_path, "config.json")
    is_hopper = check_gpu()[0]
    if (is_hopper and 'compass-13b' in model_path) or (not os.path.exists(config_file)):
        print(f"Missing {config_file}, starting quantization...")
        cmd = [
            "bash", "/workspace/scripts/convert_compress_model.sh",
            str(model_path), "hf", str(precision), device_id
        ]
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(model_path.rstrip("/")), text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Quantize failed: {quant_model_path}")
        # verify
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Quantize failed: {quant_model_path}")
    quant_model_path = quant_model_path + '/'  # 避免1.0.0结尾的模型加载错误
    print(f"Quantize completed to {quant_model_path}")
    return quant_model_path


# gererate UT dynamiclly
def pytest_generate_tests(metafunc):
    models = metafunc.config.model_list
    if "model_name" in metafunc.fixturenames and "model_config" in metafunc.fixturenames:
        if not models:
            pytest.fail("No models found for testing")
        metafunc.parametrize(
            "model_name, model_config",
            models,
            ids=[name for name, _ in models]
        )

ONELLM_TEST_API = str(os.getenv("ONELLM_TEST_API", "generate")).strip("/")

def get_model_list(api_url: str="http://0.0.0.0:23333/v1/models"):
    """Get model list from api server."""
    response = requests.get(api_url)
    if hasattr(response, 'text'):
        model_list = json.loads(response.text)
        model_list = model_list.pop('data', [])
        return [item['id'] for item in model_list]
    return None
    
def post_request(prompt, image_url=None):
    while ONELLM_TEST_API in ["v1/chat/completions", "v1/completions"]:
        model_name = get_model_list("http://0.0.0.0:23333/v1/models")[0]
        content = [{'type': 'text', 'text': prompt}]
        if image_url:
            content.append({'type': 'image_url', 'image_url': {'url': image_url}})
        messages=[{
            'role': 'user',
            'content': content,
        }]
        if ONELLM_TEST_API == "v1/chat/completions":
            pload = {
                'messages': messages,
                'model': model_name,
                'stream': False,
                'do_sample': False,
                'top_k': 1,
                'output_len': 64,
                'do_preprocess': True,
                'traceid': 123457,
            }
            res = requests.post(url="http://0.0.0.0:23333/v1/chat/completions", 
                                json=pload, 
                                headers = {'content-type': 'application/json'},
                                stream=False)
            return res
        elif ONELLM_TEST_API == "v1/completions":
            if image_url:
                break
            pload = {
                'prompt': prompt,
                'model': model_name,
                'stream': False,
                'do_sample': False,
                'top_k': 1,
                'output_len': 64,
                'do_preprocess': True,
                'traceid': 123458,
            }
            res = requests.post(url="http://0.0.0.0:23333/v1/completions", 
                                json=pload, 
                                headers = {'content-type': 'application/json'},
                                stream=False)
            return res
    data = {
        'question': prompt,
        'do_sample': False,
        'top_k': 1,
        'output_len': 64,
        'do_preprocess': True
    }
    if image_url:
        data['image_url'] = image_url
    input_dict = {
        'traceid': 123456,
        'data': data
    }
    res = requests.post(url="http://0.0.0.0:23333/generate", data=json.dumps(input_dict))
    return res


@pytest.fixture
def server_runner(model_config):
    # 量化模型（如有）
    is_hopper = check_gpu()[0]
    model_path = model_config.model_path + '/'  # 避免1.0.0结尾的模型加载错误
    if model_config.quantization:
        quant_method = model_config.quantization
        if "fp8" in quant_method:
            assert is_hopper, "This test requires a Hopper architecture GPU"
        model_path = quantize_model(model_config.model_path, quant_method, model_config.service_args)

    # 启动服务
    server_runner = ServerRunner(model_path, model_config.service_args)

    # 返回 server_runner
    yield server_runner

    # 测试结束后关闭服务并释放资源
    print("\nTeardown server_runner...")
    server_runner.terminate_and_wait()
    del server_runner
    gc.collect()


def test_server(model_name, model_config, server_runner):
    print(f"\nTesting model: {model_name}\nModel_config: {model_config}")
    # 运行测试用例
    for test_data in model_config.test_data:
        res = post_request(test_data.prompt, test_data.image_url)
        assert res.status_code == 200, f"Request failed with status code {res.status_code}"

        response = res.json()['text']
        expected_answer = test_data.ground_truth
        if test_data.verification == "==":
            assert response == expected_answer, f"Expected '{expected_answer}', got '{response}'"
        elif test_data.verification == "in":
            assert expected_answer.lower() in response.lower(), f"Expected '{expected_answer}' to be in '{response}'"
        elif test_data.verification == "startswith":
            assert response.lower().strip().startswith(expected_answer.lower().strip()), f"Expected '{response}' start with '{expected_answer}'"
        else:
            pytest.fail(f"Unknown verification method: {test_data.verification}")
