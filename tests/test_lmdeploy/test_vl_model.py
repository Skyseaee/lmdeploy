################################################################################
# @Copyright: 2019-2025 Shopee. All Rights Reserved.
# @Author   : wenlong.cao@shopee.com
# @Date     : 2025-03-05 15:15:15
# @Details  : OneLLM support VLM inference test case
################################################################################
import os
import unittest
from unittest.mock import patch
import torch

try:
    from parameterized import parameterized_class
except ImportError:
    import sys
    os.system(f"{sys.executable} -m pip install parameterized==0.9.0")
    os.system(f"{sys.executable} -m pip uninstall -y flash_attn")
    try:
        from parameterized import parameterized_class
    except ImportError:
        raise ImportError(
            'please install parameterized by pip install parameterized'
        )
    
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/workspace/mnt")

def init_params(model_dir):
    model_list = {
       # "compassllvm-v1.6": f"{model_dir}/CompassLLVM_v1_6/",
        "compassllvm_1.6": f"{model_dir}/compassvl-1.6.0/",
        "qwen2.5vl-7b": f"{model_dir}/Qwen2.5-VL-7B-Instruct/"
    }
    
    messages = [
        dict(role='user', content=[
            dict(type='text', text="You are an AI assistent. Give me a short description for the image."),
            dict(type='image_url', image_url=dict(url='https://cf.shopee.sg/file/vn-11134207-7qukw-lgbyq7x8fbav0b_tn')),
        ])
    ]
    test_class_params = []
    for model_name, model_path in model_list.items():
        # 'model_name', 'model_path', 'messages', 'max_tokens', 'cache_max_entry_count', 'vision_batchsize', 'extra_args'
        test_class_params.append((model_name, model_path, messages, 256, 0.5, 1, {"tp": 1}))
    return test_class_params 

@unittest.skipUnless(torch.cuda.device_count() > 0, "Skipping test: Requires GPU")
@unittest.skipUnless(torch.cuda.get_device_capability("cuda") >= (8, 0), "GPU Capability >= 8.0")
@parameterized_class(('model_name', 'model_path', 'messages', 'max_tokens', 'cache_max_entry_count', 'vision_batchsize', 'extra_args'), init_params(AIP_MODEL_DIR))
class TestMLLMBasic(unittest.TestCase):
    def setUp(self):
        from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline, VisionConfig
        self.session_len = 8192
        self.pipe = pipeline(self.model_path,
                    backend_config=TurbomindEngineConfig(cache_max_entry_count=self.cache_max_entry_count,
                                                         session_len=self.session_len,
                                                         **self.extra_args),
                    log_level="ERROR", max_log_len=0,
                    vision_config=VisionConfig(max_batch_size=self.vision_batchsize))
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=0.99,
            top_k=3,
            temperature=0.1,
            repetition_penalty=1.05,
        )
        self.gen_config = GenerationConfig(**gen_kwargs)

    def tearDown(self):
        del self.pipe
        torch.cuda.empty_cache()

    def test_inference(self):
        self.assertIsNotNone(self.pipe, msg=f"{self.model_name}: {self.model_path} init failed")
        
        out = self.pipe(self.messages, gen_config=self.gen_config)
        self.assertGreater(out.input_token_len, 0)
        self.assertGreater(out.generate_token_len, 0, msg=f'output:{out.text}')
        self.assertLessEqual(out.generate_token_len + out.input_token_len, self.session_len, 
                             msg=f"input+output tokens: {out.generate_token_len + out.input_token_len} > {self.session_len}")
        self.assertLessEqual(out.generate_token_len, self.max_tokens, 
                             msg=f"gen tokens:{out.generate_token_len} > {self.max_tokens}")
        self.assertIsNotNone(out.text, 
                             msg=f"{out.text} shouldn't be none")
        print(f"✨[{self.id()}][model_path:{self.model_path}][sampling:{self.gen_config}][out:{out.text}]")

@unittest.skipUnless(torch.cuda.device_count() >= 2, "Skipping test: Requires GPUs >= 2")
@unittest.skipUnless(torch.cuda.get_device_capability("cuda") >= (8, 0) or torch.cuda.get_device_capability("cuda") == (7, 0), "GPU Capability >= 8.0 or ==7.0")
@parameterized_class(('model_name', 'model_path', 'messages', 'max_tokens', 'cache_max_entry_count', 'vision_batchsize', 'extra_args'), init_params(AIP_MODEL_DIR))
class TestMLLM2xGPUs(unittest.TestCase):
    def setUp(self):
        from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline, VisionConfig
        self.session_len = 8192
        self.extra_args.update({"tp":2})
        self.pipe = pipeline(self.model_path,
                    backend_config=TurbomindEngineConfig(cache_max_entry_count=self.cache_max_entry_count,
                                                         session_len=self.session_len,
                                                         **self.extra_args),
                    log_level="ERROR", max_log_len=0,
                    vision_config=VisionConfig(max_batch_size=self.vision_batchsize, instance_num=2))
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=0.99,
            top_k=3,
            temperature=0.1,
            repetition_penalty=1.05,
        )
        self.gen_config = GenerationConfig(**gen_kwargs)

    def tearDown(self):
        del self.pipe
        torch.cuda.empty_cache()
    
    def test_inference(self):
        self.assertIsNotNone(self.pipe, msg=f"{self.model_name}: {self.model_path} init failed")
        
        out = self.pipe(self.messages, gen_config=self.gen_config)
        self.assertGreater(out.input_token_len, 0)
        self.assertGreater(out.generate_token_len, 0, msg=f'output:{out.text}')
        self.assertLessEqual(out.generate_token_len + out.input_token_len, self.session_len, 
                             msg=f"input+output tokens: {out.generate_token_len + out.input_token_len} > {self.session_len}")
        self.assertLessEqual(out.generate_token_len, self.max_tokens, 
                             msg=f"gen tokens:{out.generate_token_len} > {self.max_tokens}")
        self.assertIsNotNone(out.text, 
                             msg=f"{out.text} shouldn't be none")
        print(f"✨[{self.id()}][model_path:{self.model_path}][sampling:{self.gen_config}][out:{out.text}]")
        

@unittest.skipUnless(torch.cuda.device_count() >= 4, "Skipping test: Requires GPUs >= 4")
@parameterized_class(('model_name', 'model_path', 'messages', 'max_tokens', 'cache_max_entry_count', 'vision_batchsize', 'extra_args'), init_params(AIP_MODEL_DIR))
class TestMLLMM4xGPUs(unittest.TestCase):
    def setUp(self):
        from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline, VisionConfig
        self.session_len = 4096
        self.extra_args.update({"max_batch_size":4})
        self.extra_args.update({"tp":4})
        self.pipe = pipeline(self.model_path,
                    backend_config=TurbomindEngineConfig(cache_max_entry_count=self.cache_max_entry_count,
                                                         session_len=self.session_len,
                                                         **self.extra_args),
                    log_level="ERROR", max_log_len=0,
                    vision_config=VisionConfig(max_batch_size=self.vision_batchsize, instance_num=2))
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=False,
            top_p=0.99,
            top_k=3,
            temperature=0.1,
            repetition_penalty=1.05,
        )
        self.gen_config = GenerationConfig(**gen_kwargs)

    def tearDown(self):
        del self.pipe
        torch.cuda.empty_cache()
    
    def test_inference(self):
        self.assertIsNotNone(self.pipe, msg=f"{self.model_name}: {self.model_path} init failed")
        
        out = self.pipe(self.messages, gen_config=self.gen_config)
        self.assertGreater(out.input_token_len, 0)
        self.assertGreater(out.generate_token_len, 0, msg=f'output:{out.text}')
        self.assertLessEqual(out.generate_token_len + out.input_token_len, self.session_len, 
                             msg=f"input+output tokens: {out.generate_token_len + out.input_token_len} > {self.session_len}")
        self.assertLessEqual(out.generate_token_len, self.max_tokens, 
                             msg=f"gen tokens:{out.generate_token_len} > {self.max_tokens}")
        self.assertIsNotNone(out.text, 
                             msg=f"{out.text} shouldn't be none")
        print(f"✨[{self.id()}][model_path:{self.model_path}][sampling:{self.gen_config}][out:{out.text}]")


if __name__ == "__main__":
    unittest.main()