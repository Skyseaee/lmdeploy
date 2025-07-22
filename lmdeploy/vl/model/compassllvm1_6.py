#-*- encoding:utf-8-*-
# @Copyright: 2025 Shopee. All Rights Reserved.
# @File: compassllvm.py
# @Author: wenlong.cao@shopee.com
# @Description: Used to support loading and inference of the CompassLLVM model
# @Update History:
# Version 1.0(2024-11): Based on Llava, updated the LLM part to CompassLLM 13B
# Version 1.6(2025-04-15): Based on 1.0, built with InternVL-V2.5 and CompassLLM 13B
# Version 2.0(2025-08-15): Based on 1.6, updated the LLM part: CompassLLM 13B -> CompassLLM-SMoE
#
import warnings
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL.Image import Image
from transformers import AutoConfig, AutoModel
from torchvision.transforms import v2 as transforms

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')

from lmdeploy.vl.model.onepiece.utils import is_support_optimize_vlm, gen_input_dict, trt_version
VLM_ENABLE_TRT = os.environ.get("ONELLM_VLM_ENABLE_TRT", False) and is_support_optimize_vlm()
if VLM_ENABLE_TRT:
    from lmdeploy.vl.model.onepiece.workflow import build_engine


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=12,
                       image_size=448,
                       use_thumbnail=False):
    """copy from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    logger.info(f"[DHR({min_num},{max_num},{image_size},{use_thumbnail})] {orig_width}x{orig_height} -> {target_width}x{target_height} -> {blocks} images -> {len(processed_images)*256} tokens")
    return processed_images


class ImageEncoderWrapperV1d6(object):
    """TensorRT Builder for VisionEncoder
    """
    def __init__(self, vlm, config):
        super().__init__(vlm, config)
        self.model = vlm.vision_model
        self.mlp1 = vlm.mlp1
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

    def get_cfg(self, config, max_batch_size):
        IMAGE_SIZE = config.image_size
        model_name = os.path.join(os.path.dirname(__file__), f"compassllvm_v1.6_vit_bz{max_batch_size}_v{trt_version()}")
        cfg = {
            'onnx_path': f'{model_name}.onnx',
            'trt_path': f'{model_name}.engine',
            'input_names': ['pixel_values'],
            'input_shapes': [[1, 3, IMAGE_SIZE, IMAGE_SIZE]],
            'input_dtypes': [torch.float16],
            'output_names': ['hidden_states'],
            'dynamic_axes': {'pixel_values': {0: 'batch_size'},'hidden_states': {0: 'batch_size'}},
            'min_input_shapes': [[1, 3, IMAGE_SIZE, IMAGE_SIZE]],
            'opt_input_shapes': [[max_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE]],
            'max_input_shapes': [[max_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE]],
            'precision': 'fp16',
            'max_workspace_size': 40*2**30,
            'validate_method': 'cosine_distance',
        }
        cfg["input_dict"] = gen_input_dict(cfg["input_names"], cfg["input_shapes"], cfg["input_dtypes"])
        return cfg

    def post_process(self, vit_embeds):
        def pixel_shuffle(x, scale_factor=0.5):
            """Pixel shuffle for image feature map.
            N, W, H, C --> N, W, H * scale, C // scale
            N, W, H * scale, C // scale --> N, H * scale, W, C // scale
            N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
            """
            n, w, h, c = x.size()
            x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                    int(c / (scale_factor * scale_factor)))
            # if self.ps_version == 'v1':
            #     warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
            #                 'which results in a transposed image.')
            # else:
            x = x.permute(0, 2, 1, 3).contiguous()
            return x
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values).last_hidden_state
        return self.post_process(outputs)
    

@VISION_MODELS.register_module()
class CompassLLVM_V1d6(VisonModel):
    _arch = 'CompassLLVM'
    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = '',
                 default_device="auto"):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)
        self.default_device = default_device

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == cls._arch and hasattr(config, 'llm_config') and hasattr(config, 'vision_config'):
            if hasattr(config.llm_config, "num_experts"):
                setattr(config, "version", "2.0")
            else:
                setattr(config, "version", "1.6")
            return True
        return False

    def build_preprocessor(self):
        input_size = self.hf_config.vision_config.image_size
        # TODO(cwl): refactor for v1.0 and v1.6 use same preprocess func
        # image preprocessing
        resample_dict = {
            0: transforms.InterpolationMode.NEAREST,
            1: transforms.InterpolationMode.NEAREST_EXACT,
            2: transforms.InterpolationMode.BILINEAR,
            3: transforms.InterpolationMode.BICUBIC,
        }
        # if hasattr(self.hf_config.vision_config, "resample"):
        #     resample_mode = resample_dict[int(self.hf_config.vision_config.resample)]
        # else:
        resample_mode = transforms.InterpolationMode.BILINEAR

        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')
                        if img.mode != 'RGB' else img),
            transforms.Resize((input_size, input_size),
                        interpolation=resample_mode),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        
    def build_model(self):
        """Load model."""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights():
            self.model_dtype = self.hf_config.torch_dtype
            if VLM_ENABLE_TRT:
                self.model_dtype = torch.float16
            config = self.hf_config
            model = AutoModel.from_config(config, trust_remote_code=True)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model
            model.to(dtype=self.model_dtype)

        with disable_logging():
            if self.with_llm:
                model.language_model.tie_weights()
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=self.default_device if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=['InternVisionEncoderLayer'],
                dtype=self.model_dtype)

        # We need eval mode to freeze the weights in model, thus,
        # avoid randomness in inference.
        self.model = model.eval()
        self.config = config

        if VLM_ENABLE_TRT:
            logger.warning("✨CompassLLVM v1.6 enable_image_trt")
            self.vision_model = ImageEncoderWrapperV1d6(self.model, self.hf_config)
            self.vision_batch_size = max(int(os.environ.get("ONELLM_TRT_VISION_MAX_BATCH_SIZE", "1")), 16)
            cfg = self.vision_model.get_cfg(self.hf_config.vision_config, self.vision_batch_size)
            self.trt_vision_model = build_engine(cfg=cfg, model=self.vision_model, device=self.vision_model.device)
            del self.vision_model.model
            os.system(f"rm -rf {cfg['onnx_path']}")
        else:
            logger.warning("✨CompassLLVM v1.6 enable_image_torch")
            self.vision_model = self.model.vision_model

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        image_res = {'low': 1, 'medium': 6, 'high': 12}
        outputs = []
        images = self.collect_images(messages)
        for image, param in images:
            max_num = param.get('max_dynamic_patch')
            if max_num is None or not isinstance(max_num, int):
                res_key = param.get('detail', 'default')
                max_num = image_res.get(res_key, self.config.max_dynamic_patch)
            
            image = image.convert('RGB')
            patch_images = dynamic_preprocess(
                image,
                min_num=self.config.min_dynamic_patch,
                max_num=max_num,
                image_size=self.config.vision_config.image_size,
                use_thumbnail=self.config.use_thumbnail)
        
            out = [self.image_transform(x) for x in patch_images]
            out = torch.stack(out)  # (patch) x c x h x w
            outputs.append(dict(pixel_values=out, 
                                patch_nums=len(patch_images),
                                image_size=image.size,
                                image_tokens=self.model.num_image_token,
                                image_token_id=self.image_token_id))
            
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """forward."""
        images = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        split = [x["patch_nums"] for x in images]
        outputs = torch.cat([x["pixel_values"] for x in images], dim=0).to(self.model.device, dtype=self.model.dtype)
        logger.info(f"CompassLLVM: image={len(images)}, split={split}, vision_input={outputs.shape}")
        if VLM_ENABLE_TRT:
            B = outputs.shape[0]
            if B > max_batch_size:
                hs = []
                for i in range(0, B, max_batch_size):
                    start_idx = i
                    end_idx = i + max_batch_size
                    tmp_hs = self.trt_vision_model({"pixel_values": outputs[start_idx:end_idx]})
                    hs.append(tmp_hs)
                embedding_outputs = torch.cat(hs, dim=0)
            else:
                embedding_outputs = self.trt_vision_model({"pixel_values":outputs})
        else:
            embedding_outputs = self.model.extract_feature(outputs)

        embedding_outputs = torch.split(embedding_outputs, split, dim=0)
        embedding_outputs = [x.reshape(-1, x.shape[-1]) for x in embedding_outputs]
        if self.model.dtype == torch.bfloat16:
            embedding_outputs = [x.float() for x in embedding_outputs]
        messages.append(dict(role='forward', content=embedding_outputs))
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

def compassllvm_ut():
    for max_bz in [1]:
        print(f"max_bz={max_bz}")
        os.environ["ONELLM_VLM_ENABLE_TRT"] = "1"
        os.environ["ONELLM_TRT_VISION_MAX_BATCH_SIZE"] = str(max_bz)
        from transformers import AutoConfig
        model_dir = "/data/models/Compassllvm_V1_6_pre"
        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = CompassLLVM_V1d6(model_path=model_dir, with_llm=False, hf_config=hf_config)
        model.build_model()
        print(f"build max_bz={max_bz} model done")
        if hf_config.version == "1.6":
            cfg = model.vision_model.get_cfg(hf_config.vision_config, max_bz)
        else:
            cfg = model.vision_model.get_cfg(hf_config.visual_tokenizer_config, max_bz)
        input_dict = {}
        for k, v in cfg['input_dict'].items():
            input_dict[k] = v.cuda()
        ret = model.trt_vision_model(input_dict)
        print(ret.shape)
        print(ret.dtype)
        print(ret)

if __name__ == "__main__":
    compassllvm_ut()