#-*- encoding:utf-8-*-
# @Copyright: 2025 Shopee. All Rights Reserved.
# @File: compassllvm.py
# @Author: wenlong.cao@shopee.com
# @Description: Used to support loading and inference of the CompassLLVM model
# Version 1.0(2024-11): Based on Llava, updated the LLM part to CompassLLM 13B

import warnings
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from torchvision.transforms import v2 as transforms

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')

from lmdeploy.vl.model.onepiece.utils import is_support_optimize_vlm, gen_input_dict, trt_version
VLM_ENABLE_TRT = os.environ.get("ONELLM_VLM_ENABLE_TRT", False) and is_support_optimize_vlm()
if VLM_ENABLE_TRT:
    from lmdeploy.vl.model.onepiece.workflow import build_engine


class _ImagePreprocessor(object):
    """Image Preprocess torchvision version
    
    """
    def __init__(self, config):
        self.do_resize = config.do_resize
        self.do_rescale = config.do_rescale
        self.do_normalize = config.do_normalize
        self.image_std = config.image_std
        self.image_mean = config.image_mean
        self.size = config.size
        self.resample = {
            0: transforms.InterpolationMode.NEAREST,
            1: transforms.InterpolationMode.NEAREST_EXACT,
            2: transforms.InterpolationMode.BILINEAR,
            3: transforms.InterpolationMode.BICUBIC,
        }
        if int(config.resample) in self.resample:
            resample_mode = self.resample[int(config.resample)]
        else:
            resample_mode = transforms.InterpolationMode.BILINEAR
        # [H, W, 3] -> [384, 384, 3] -> [3, 384, 384]/255 -> Norm([3, 384, 384])
        self.transform = transforms.Compose([
            transforms.Resize([self.size["height"], self.size["width"]], interpolation=resample_mode),                   
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ])
        
    def __call__(self, image, convert_to_rgb=True):
        """For performance, we may ignore the tail 1-pixel in row/col"""
        if convert_to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        width, height = center_x*2, height*2
        # ----- IO
        image_top_left = image.crop((0, 0, center_x, center_y))
        image_top_right = image.crop((center_x, 0, width, center_y))
        image_bottom_left = image.crop((0, center_y, center_x, height))
        image_bottom_right = image.crop((center_x, center_y, width, height))
        # ----- 
        # ----- Compute
        imgs = [image, image_top_left, image_top_right, image_bottom_left, image_bottom_right]
        # [H, W, 3] -> [3, 384, 384] -> [15, 384, 384] -> [1, 15, 384, 384]
        values = torch.cat([self.transform(img) for img in imgs]).unsqueeze(0)
        # -----
        return values


def post_process(n, features):
    """combined the vision encoder output features to a tensor
    
    - n:
    - featues: 
    """
    _, l, d = features.shape
    features = features.reshape(n, 5, l, d)
    features_overall = features[:, 0, :, :]  # [n, l, d]
    features_parts = features[:, 1:, :, :]  # [n, 4, l, d]
    sqrt_l = int(l ** 0.5)
    assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
    features_parts = features_parts.reshape(n, 4, sqrt_l, sqrt_l, d)  # [n, 4, sqrt(l), sqrt(l), d]
    features_top = torch.concat([features_parts[:, 0, :, :, :], features_parts[:, 1, :, :, :]],
                                dim=-2)  # [n, sqrt(l), sqrt(l)*2, d]
    features_bottom = torch.concat([features_parts[:, 2, :, :, :], features_parts[:, 3, :, :, :]],
                                    dim=-2)  # [n, sqrt(l), sqrt(l)*2, d]
    features_merge = torch.concat([features_top, features_bottom], dim=-3)  # [n, sqrt(l)*2, sqrt(l)*2, d]
    features_pool = F.interpolate(features_merge.permute(0, 3, 1, 2).to(torch.float32), size=sqrt_l,
                                    mode='area')  # [n, d, sqrt_l, sqrt_l]
    features_pool = features_pool.flatten(2).permute(0, 2, 1).to(features.dtype)  # [n, l, d]
    features = torch.cat([features_overall, features_pool], dim=-1)  # [n, l, 2*d]
    return features


class _VLImageEncoder(torch.nn.Module):
    """TensorRT Builder for VisionEncoder
    """
    def __init__(self, vision_model, config: AutoConfig=None):
        super().__init__()
        self.model = vision_model
        self.config = config
        self.dtype = vision_model.dtype
        self.device = vision_model.device

    def get_cfg(self, config, max_batch_size):
        assert hasattr(config, 'backbone_config'), "config should have backbone_config"
        
        IMAGE_SIZE = config.backbone_config.image_size
        model_name = os.path.join(os.path.dirname(__file__), f"compassllvm_vit_bz{max_batch_size}_v{trt_version()}")
        cfg = {
            'onnx_path': f'{model_name}.onnx',
            'trt_path': f'{model_name}.engine',
            'input_names': ['pixel_values'],
            'input_shapes': [[1, 15, IMAGE_SIZE, IMAGE_SIZE]],
            'input_dtypes': [torch.float16],
            'output_names': ['hidden_states'],
            'dynamic_axes': {'pixel_values': {0: 'batch_size'},'hidden_states': {0: 'batch_size'}},
            'min_input_shapes': [[1, 15, IMAGE_SIZE, IMAGE_SIZE]],
            'opt_input_shapes': [[max_batch_size, 15, IMAGE_SIZE, IMAGE_SIZE]],
            'max_input_shapes': [[max_batch_size, 15, IMAGE_SIZE, IMAGE_SIZE]],
            'precision': 'fp16',
            'max_workspace_size': 40*2**30,
            'validate_method': 'cosine_distance',
        }
        input_dict = gen_input_dict(cfg["input_names"], cfg["input_shapes"], cfg["input_dtypes"])
        cfg["input_dict"] = input_dict
        return cfg

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n, c, side, _ = pixel_values.shape
        pixel_values = pixel_values.reshape(n * 5, c // 5, side, side)
        output = self.model.backbone(pixel_values, output_hidden_states=True, return_dict=True)
        features = output.hidden_states[-1]
        return post_process(n, features)


@VISION_MODELS.register_module()
class CompassVisionModel(VisonModel):
    _arch = 'CompassLLVM'
    # def __init__(self,
    #              model_path: str,
    #              with_llm: bool = False,
    #              max_memory: Dict[int, int] = None,
    #              hf_config: AutoConfig = None,
    #              backend: str = '',
    #              default_device="auto"):
    #     super().__init__(model_path, with_llm, max_memory, hf_config, backend)
    #     """init."""
    #     self.default_device = default_device

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == cls._arch and hasattr(config, 'llm_config') and hasattr(config, 'visual_tokenizer_config'):
            setattr(config, "version", "1.0")
            return True
        return False

    def build_preprocessor(self):
        from accelerate import init_empty_weights
        model = None
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_config(self.hf_config, trust_remote_code=True)
        self.tv_preprocess_image = _ImagePreprocessor(model.visual_tokenizer.image_processor)
        self.default_preprocess_image = model.visual_tokenizer.preprocess_image
        image_size = self.hf_config.visual_tokenizer_config.backbone_config.image_size
        patch_size = self.hf_config.visual_tokenizer_config.backbone_config.patch_size
        self.n_token_per_image = (image_size // patch_size)**2

    def build_model(self):
        """build model & load weights."""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_config(self.hf_config, trust_remote_code=True)
        if not self.with_llm:
            del model.llm
        else:
            self.vl_model = model
        with disable_logging():
            if self.with_llm:
                model.llm.tie_weights()
            load_checkpoint_and_dispatch(
                model=model,
                max_memory=self.max_memory,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['CLIPEncoderLayer', 'SiglipEncoderLayer'],
                dtype=torch.half)

        model.eval()
        self.model = model

        # BUILD vision model
        if VLM_ENABLE_TRT:
            logger.warning("✨CompassLLVM enable_image_trt")
            self.vision_model = _VLImageEncoder(self.model.visual_tokenizer, self.model.visual_tokenizer.config)
            self.vision_batch_size = int(os.environ.get("ONELLM_TRT_VISION_MAX_BATCH_SIZE", "16"))
            cfg = self.vision_model.get_cfg(self.hf_config.visual_tokenizer_config, self.vision_batch_size)
            self.trt_vision_model = build_engine(
                cfg=cfg,
                model=self.vision_model,
                device=self.vision_model.device)
            del self.vision_model.model
            os.system(f"rm -rf {cfg['onnx_path']}")
        else:
            logger.warning("✨CompassLLVM enable_image_torch")
            self.vision_model = self.model.visual_tokenizer

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """dispatch with input shape, torchvision only support width == height images"""
        images = self.collect_images(messages)
        outputs = []
        for image, param in images:
            width, height = image.size
            image_preprocessor = self.tv_preprocess_image if width == height else self.default_preprocess_image
            image = image.convert('RGB')
            out = image_preprocessor(image, convert_to_rgb=True)
            outputs.append(dict(
                pixel_values=out,
                image_size=image.size,
                image_tokens=self.n_token_per_image,
                image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def _encode(self, pixel_values: torch.Tensor):
        if VLM_ENABLE_TRT:
            features = self.trt_vision_model({"pixel_values":pixel_values}, dtype=self.vision_model.dtype)
            return features
        else:
            n, c, side, _ = pixel_values.shape
            pixel_values = pixel_values.reshape(n * 5, c // 5, side, side)
            output = self.vision_model.backbone(pixel_values, output_hidden_states=True, return_dict=True)
            features = output.hidden_states[-1]
            if self.vision_model.config.drop_cls_token:
                features = features[:, 1:, :]
            return post_process(n, features)

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """forward for compassllvm s2wrapper."""
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        outputs= []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [x["pixel_values"].to(dtype=self.vision_model.dtype, device=self.vision_model.device) for x in inputs[idx:idx + max_batch_size]]
            pixel_values = torch.cat(pixel_values, dim=0)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            visual_tokens = self._encode(pixel_values)
            vte_out = self.model.vte(visual_tokens)
            image_features = torch.split(vte_out, split_size_or_sections=1, dim=0)
            outputs.extend([x.squeeze() for x in image_features])
        messages.append(dict(role='forward', content=outputs))
        return messages
    
    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [item['text'] for item in message['content'] if item['type'] == 'text']
            prompt = (IMAGE_TOKEN + '\n') * n_images + content[0]
            prompt_messages.append(dict(role=message['role'], content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)


def UT_CompassLLVM1_0():
    from transformers import AutoConfig
    from lmdeploy.vl.utils import load_image
    model_dir = "/home/wenlong.cao/models/compassllvm-v1-0/"
    os.environ["ONELLM_VLM_ENABLE_TRT"] = "1"
    max_bz = 1
    print(f"max_bz={max_bz}")
    print(f"VLM_ENABLE_TRT={VLM_ENABLE_TRT}")
    os.environ["ONELLM_TRT_VISION_MAX_BATCH_SIZE"] = str(max_bz)
    messages = [
        dict(role='user', content=[
            dict(type='text', text="You are an AI assistent. Give me a short description for the image."),
            dict(type='image', image=load_image('https://cf.shopee.sg/file/vn-11134207-7qukw-lgbyq7x8fbav0b'),
                resize_width=448, resize_height=448, cache_hit=False),
        ])
    ]
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = CompassVisionModel(model_path=model_dir, with_llm=False, hf_config=hf_config)
    model.build_model()
    model.build_preprocessor()
    messages = model.preprocess(messages)
    messages = model.forward(messages, max_batch_size=max_bz)
    print(f"messages={messages[-1]['content']}")
    
    
if __name__ == "__main__":
    UT_CompassLLVM1_0()