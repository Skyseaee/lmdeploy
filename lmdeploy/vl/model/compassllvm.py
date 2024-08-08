# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os
from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
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

class ImagePreprocessor(object):
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


class ImageEncoderWrapper(torch.nn.Module):
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


class ImageEncoderWrapperV1d6(ImageEncoderWrapper):
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
class CompassLLVM(VisonModel):
    _arch = 'CompassLLVM'
    def __init__(self, model_path: str,
                       with_llm: bool = False,
                       max_memory: Dict[int, int] = None,
                       hf_config: AutoConfig = None,
                       default_device="auto"):
        """init."""
        self.model_path = model_path
        self.with_llm = with_llm
        self.max_memory = max_memory
        self.hf_config = hf_config
        self.default_device = default_device
        self.build_model()

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == cls._arch and hasattr(config, 'llm_config') and hasattr(config, 'visual_tokenizer_config'):
            setattr(config, "version", "1.0")
            return True
        elif arch == cls._arch and hasattr(config, 'llm_config') and hasattr(config, 'vision_config'):
            setattr(config, "version", "1.6")
            return True
        return False

    def build_model(self):
        if self.hf_config.version == "1.6":
            self.build_model_v1_6()
        else:
            self.build_model_1_0()
            
    def build_model_1_0(self):
        """build model & load weights."""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_config(self.hf_config, trust_remote_code=True)
        if not self.with_llm:
            # delete compass base model meta information, we will load the LLM part at torbomind engine
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
                device_map=self.default_device if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=[],
                dtype=torch.half)

        self.model = model
        self.model.eval()
        
        self.tv_preprocess_image = ImagePreprocessor(self.model.visual_tokenizer.image_processor)
        self.default_preprocess_image = self.model.visual_tokenizer.preprocess_image
        if VLM_ENABLE_TRT:
            logger.warning("✨CompassLLVM enable_image_trt")
            self.vision_model = ImageEncoderWrapper(self.model.visual_tokenizer, self.model.visual_tokenizer.config)
            self.vision_batch_size = int(os.environ.get("ONELLM_TRT_VISION_MAX_BATCH_SIZE", "32"))
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

    def encode(self, pixel_values: torch.Tensor):
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

    def dispatch_preprocess_image(self, image):
        """dispatch with input shape, torchvision only support width == height images"""
        width, height = image.size
        if width == height:
            return self.tv_preprocess_image(image, convert_to_rgb=True)
        else:
            return self.default_preprocess_image(image, convert_to_rgb=True)

    def preprocess(self, images: List[Image]) -> Union[torch.Tensor, List[torch.Tensor]]:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.dispatch_preprocess_image, image) for image in images]
            image_tensor_list = [future.result() for future in as_completed(futures)]
        return image_tensor_list

    def forward_v1_0(self, images: List[Image]) -> List[torch.Tensor]:
        """forward for compassllvm s2wrapper."""
        pixel_values = self.preprocess(images)
        with torch.no_grad():
            num_images = [x.shape[0] for x in pixel_values]
            pixel_values = [x.to(dtype=self.vision_model.dtype, device=self.vision_model.device) for x in pixel_values]
            visual_tokens = self.encode(torch.cat(pixel_values, dim=0))
            vte_out = self.model.vte(visual_tokens)
            visual_embeds = torch.split(vte_out, split_size_or_sections=num_images, dim=0)
        return visual_embeds


    ############################## V1.6 ##############################
    def build_model_v1_6(self):
        """Load model."""
        from accelerate import init_empty_weights
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

        from accelerate import load_checkpoint_and_dispatch
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
        input_size = self.config.vision_config.image_size
        # TODO(cwl): refactor for v1.0 and v1.6 use same preprocess func
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')
                        if img.mode != 'RGB' else img),
            transforms.Resize((input_size, input_size),
                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
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

    def preprocess_v1_6(self, images: List[Image], params: List[Dict] = None):
        if params is not None:
            assert len(images) == len(
                params), 'different length of images and params'
        else:
            params = [{}] * len(images)

        image_res = {'low': 1, 'medium': 6, 'high': 12}

        outputs = []
        for image, param in zip(images, params):
            max_num = param.get('max_dynamic_patch')
            if max_num is None or not isinstance(max_num, int):
                res_key = param.get('detail', 'default')
                max_num = image_res.get(res_key, self.config.max_dynamic_patch)
            out = dynamic_preprocess(
                image,
                min_num=self.config.min_dynamic_patch,
                max_num=max_num,
                image_size=self.config.vision_config.image_size,
                use_thumbnail=self.config.use_thumbnail)
            out = [self.transform(x) for x in out]
            out = torch.stack(out)  # (patch) x c x h x w
            outputs.append(out)
        return outputs

    @torch.no_grad()
    def forward_v1_6(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        """forward."""
        images = [x.convert('RGB') for x in images]
        outputs = self.preprocess_v1_6(images, params)
        split = [x.shape[0] for x in outputs]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.to(self.model.device, dtype=self.model.dtype)
        logger.info(f"CompassLLVM: image={len(images)}, split={split}, vision_input={outputs.shape}")
        if VLM_ENABLE_TRT:
            B = outputs.shape[0]
            if B > self.vision_batch_size:
                hs = []
                for i in range(0, B, self.vision_batch_size):
                    start_idx = i
                    end_idx = i + self.vision_batch_size
            
                    input_dicts = {
                        "pixel_values": outputs[start_idx:end_idx]
                    }
                    tmp_hs = self.trt_vision_model(input_dicts)
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
        return embedding_outputs
    
    ############################## V1.6 ##############################
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        if self.hf_config.version == "1.6":
            return self.forward_v1_6(images, params)
        else:
            return self.forward_v1_0(images)
    
def compassllvm_ut():
    for max_bz in [1]:
        print(f"max_bz={max_bz}")
        os.environ["ONELLM_VLM_ENABLE_TRT"] = "1"
        os.environ["ONELLM_TRT_VISION_MAX_BATCH_SIZE"] = str(max_bz)
        from transformers import AutoConfig
        model_dir = "/data/models/Compassllvm_V1_6_pre"
        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = CompassLLVM(model_path=model_dir, with_llm=False, hf_config=hf_config)
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