#-*- encoding: utf-8 -*-
"""
Support Qwen2.5-VL-7B-Instruct model using turbomind
Author: wenlong.cao@shopee.com
Date: 2025-02-24 17:00:12
"""
import sys
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.onepiece.utils import device_default_half_type

logger = get_logger('lmdeploy')


def check_qwen_2d5_vl_deps_install():
    """check qwen_vl_utils."""
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        os.system(f"{sys.executable} -m pip install qwen_vl_utils")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        try:
            import qwen_vl_utils  # noqa: F401
        except ImportError:
            raise ImportError(
                'please install qwen_vl_utils by pip install qwen_vl_utils'  # noqa: E501
            )

from lmdeploy.vl.model.onepiece.utils import is_support_optimize_vlm, trt_version

VLM_ENABLE_TRT = os.environ.get("ONELLM_VLM_ENABLE_TRT", False) and is_support_optimize_vlm()
if VLM_ENABLE_TRT:
    from lmdeploy.vl.model.onepiece.workflow import build_engine


class _ImageEncoderWrapper(torch.nn.Module):
    """TensorRT Builder for VisionEncoder
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().cuda()
        self.device = model.device
        self.dtype = model.dtype

    def get_cfg(self, processor, max_batch_size=1, attn_implement="eager", model_path="None"):
        """For a fixed input size images, we use dynamic batchsize else use dynamic input shape
        Note:
        Read Fixed input size from environments ``ONELLM_IMAGE_SIZE`` format is "320x320" -> 308x308
        """
        config = processor.image_processor
        IMAGE_FACTOR = 28
        input_shapes = os.environ.get("ONELLM_IMAGE_SIZE", "512x512")
        rotary_pos_emb_dim = self.model.config.hidden_size // self.model.config.num_heads // 2
        try:
            width, height = int(input_shapes.split("x")[0]), int(input_shapes.split("x")[1])
            from lmdeploy.vl.utils import load_image
            from qwen_vl_utils import smart_resize, process_vision_info
            (resized_width, resized_height) = smart_resize(height, width, factor=IMAGE_FACTOR,
                                                min_pixels=config.min_pixels, max_pixels=config.max_pixels) 

            image_urls = ["https://cf.shopee.sg/file/vn-11134207-7qukw-lgbyq7x8fbav0b",
                            "https://cf.shopee.sg/file/cafd5178e11af1aa5152da00072b387c",
                            "https://cf.shopee.sg/file/0e23522704afef7058a559135d8f0b1f",
                            "https://cf.shopee.sg/file/0a28ee4750b466f6ac45ac6ba8193de9",
                            "https://cf.shopee.sg/file/vn-11134207-7r98o-lxw0rvnh5gwr3d",
                            "https://cf.shopee.sg/file/sg-11134202-7rd5y-luspn91mzfch9c",
                            "https://cf.shopee.sg/file/sg-11134201-7rbk9-m5ikdn2j5sevea",
                            "https://cf.shopee.sg/file/my-11134207-7r990-lotihhuuq5cgd1"]*max_batch_size
            images = [load_image(url) for url in image_urls[:max_batch_size]]
            images = [x.convert('RGB') for x in images]
            content = []
            for image in images:
                item = dict(type='image', image=image)
                item.update({"resized_height": height, "resized_width": width})
                content.append(item)
            messages = [dict(content=content)]
            image_inputs, _ = process_vision_info(messages)
            image_inputs = processor.image_processor(images=image_inputs, videos=None, return_tensors='pt')
            pixel_values = image_inputs['pixel_values'].to(dtype=torch.float16, device='cuda')
            image_grid = image_inputs['image_grid_thw'].to(dtype=torch.int32, device='cuda')
            hidden_states, rotary_pos_emb, attention_mask, cu_attention_mask, window_index = self.prepare_input(pixel_values, image_grid, attn_implement)
            L, H = hidden_states.shape
            logger.info(f"✨OneLLM: input shape {width}x{height}->{resized_width}x{resized_height}->{L}x{H}")
            print(f"🔔H={H}, L={L}")
            print(f"🔔grid_thw: {image_grid}")
            print(f"🔔pixel_values: {hidden_states.shape}, {hidden_states.dtype}")
            print(f"🔔rotary_pos_emb: {rotary_pos_emb.shape}, {rotary_pos_emb.dtype}")
            print(f"🔔attention_mask: {attention_mask.shape}, {attention_mask.dtype}")
            print(f"🔔cu_attention_mask: {cu_attention_mask.shape}, {cu_attention_mask.dtype}")
        except TypeError:
            logger.warning(f"✨OneLLM: invalid input shape {input_shapes}")
            width, height = -1, -1
        model_name = os.path.basename(model_path.strip().rstrip('/'))
        model_name = os.path.join(os.path.dirname(__file__), 
                                  f"{model_name}_{max_batch_size}x{height}x{width}_L{L}x{H}_{attn_implement}_{trt_version()}")

        cfg = {
            'onnx_path': f'{model_name}.onnx',
            'trt_path': f'{model_name}.engine',
            'input_names': ['pixel_values', 'rotary_pos_emb', 'attention_mask', 'cu_attention_mask'],
            'input_shapes': [[max_batch_size*L, H], [max_batch_size*L, rotary_pos_emb_dim], [1, max_batch_size*L, max_batch_size*L], [1, max_batch_size*L, max_batch_size*L]],
            'input_dtypes': [torch.float16, torch.float16, torch.float16, torch.float16] if attn_implement == "eager" else [torch.float16, torch.float16, torch.bool, torch.bool], 
            'output_names': ['hidden_states'],
            'dynamic_axes': None,
            'min_input_shapes': None,
            'opt_input_shapes': None,
            'max_input_shapes': None,
            'precision': 'fp16',
            'do_constant_folding': False,
            'max_workspace_size': 80*2**30,
            'validate_method': 'cosine_distance',
            'opset_version': 17
        }
        cfg["input_dict"] = {
            "pixel_values": hidden_states.cuda(),
            "rotary_pos_emb": rotary_pos_emb.cuda(),
            "attention_mask": attention_mask.cuda(),
            "cu_attention_mask": cu_attention_mask.cuda(),
        }
        return cfg

    def prepare_input(self, hidden_states, grid_thw, attn_implement='eager'):
        hidden_states = self.model.patch_embed(hidden_states)
        rotary_pos_emb = self.model.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.model.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=torch.device("cuda"),
            dtype=grid_thw.dtype,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.model.spatial_merge_unit, self.model.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.model.spatial_merge_unit, self.model.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        # for SPDA
        if attn_implement == 'sdpa':
            attention_mask = torch.zeros([1, seq_len, seq_len], device=hidden_states.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
            
            window_attention_mask = torch.zeros([1, seq_len, seq_len], device=hidden_states.device, dtype=torch.bool)
            for i in range(1, len(cu_window_seqlens)):
                window_attention_mask[..., cu_window_seqlens[i - 1] : cu_window_seqlens[i], cu_window_seqlens[i - 1] : cu_window_seqlens[i]] = True

        elif attn_implement == "eager":
            ##  for eager
            attention_mask = torch.full(
                    [1, seq_len, seq_len], torch.finfo(torch.float32).min, device=torch.device('cuda'), dtype=torch.float32
                )
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
            
            window_attention_mask = torch.full(
                    [1, seq_len, seq_len], torch.finfo(torch.float32).min, device=torch.device('cuda'), dtype=torch.float32
                )
            for i in range(1, len(cu_window_seqlens)):
                window_attention_mask[..., cu_window_seqlens[i - 1] : cu_window_seqlens[i], cu_window_seqlens[i - 1] : cu_window_seqlens[i]] = 0
        else:
            logger.error("flash_attn and tensorrt conflict, please run pip3 uninstall flash_attn before use tensorrt inference")
        return hidden_states, rotary_pos_emb, attention_mask, window_attention_mask, window_index

    def forward(self, hidden_states:torch.Tensor, 
                rotary_pos_emb:torch.Tensor, 
                attention_mask:torch.Tensor, 
                window_attention_mask:torch.Tensor
                ) -> torch.Tensor:
        #[7,15,23,31]
        # print("🌟🌟 Blocks", len(self.model.blocks))
        if self.model.fullatt_block_indexes != [7,15,23,31]:
            logger.error("only support fullatt_block_indexes = [7,15,23,31]")
            return None
        for blk in self.model.blocks[:7]:
            hidden_states = blk(hidden_states, attention_mask=window_attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = self.model.blocks[7](hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

        for blk in self.model.blocks[8:15]:
            hidden_states = blk(hidden_states, attention_mask=window_attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = self.model.blocks[15](hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        
        for blk in self.model.blocks[16:23]:
            hidden_states = blk(hidden_states, attention_mask=window_attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = self.model.blocks[23](hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        
        for blk in self.model.blocks[24:31]:
            hidden_states = blk(hidden_states, attention_mask=window_attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = self.model.blocks[31](hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        
        hidden_states = self.model.merger(hidden_states)
        return hidden_states

@VISION_MODELS.register_module()
class Qwen2d5VLModel(VisonModel):
    """Qwen2.5 VL model."""

    _arch = 'Qwen2_5_VLForConditionalGeneration'
    
    def build_preprocessor(self):
        check_qwen_2d5_vl_deps_install()
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(image_token)[-1]

    def build_model(self):
        check_qwen_2d5_vl_deps_install()

        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        if self.hf_config.torch_dtype != self.hf_config.vision_config.torch_dtype:
            logger.warning(f"🔔Qwen2_5_VLForConditionalGeneration: {self.hf_config.torch_dtype},"+ 
                           f"vision_model: {self.hf_config.vision_config.torch_dtype}")
            if self.hf_config.vision_config.torch_dtype == torch.float32:
                self.hf_config.vision_config.torch_dtype = device_default_half_type()
            else:
                self.hf_config.torch_dtype = device_default_half_type()
        
        if self.hf_config.tie_word_embeddings and self.with_llm:
            from lmdeploy.vl.model.onepiece.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration._from_config(self.hf_config, 
                                                                    torch_dtype=self.hf_config.torch_dtype, 
                                                                    attn_implementation=self.hf_config._attn_implementation) 
            self.vl_model = model
        else:
            with init_empty_weights():
                config = self.hf_config
                config.quantization_config = {}  # disable vision part quantization
                # disable accelerate check_tied_parameters_in_config
                # for Qwen2.5-VL-2B-Instruct
                config.tie_word_embeddings = False
                if VLM_ENABLE_TRT:
                    if int(os.environ.get("ONELLM_TRT_VISION_MAX_BATCH_SIZE", 1)) <= 2:
                        config._attn_implementation = "eager"
                    else:
                        config._attn_implementation = "sdpa"
                from lmdeploy.vl.model.onepiece.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
                model = Qwen2_5_VLForConditionalGeneration._from_config(config, 
                                                                        torch_dtype=config.torch_dtype, 
                                                                        attn_implementation=config._attn_implementation) 
            if not self.with_llm:
                del model.model
                del model.lm_head
            else:
                self.vl_model = model

            with disable_logging():
                load_checkpoint_and_dispatch(
                    model=model,
                    checkpoint=self.model_path,
                    device_map=self.default_device if not self.with_llm else {'': 'cpu'},
                    max_memory=self.max_memory,
                    no_split_module_classes=['Qwen2_5_VisionPatchEmbed', 'Qwen2_5_VLVisionBlock', 'Qwen2_5_VLPatchMerger'],
                    dtype=torch.half)

        self.model = model.eval()
        logger.info(f"🔔final model dtype: {self.model.dtype}")

        if VLM_ENABLE_TRT:
            logger.debug("✨Qwen2.5VL enable_image_trt")
            self.vision_batch_size = int(os.environ.get("ONELLM_TRT_VISION_MAX_BATCH_SIZE", 1))
            vision_model = _ImageEncoderWrapper(self.model.visual)
            cfg = vision_model.get_cfg(self.processor, 
                                       max_batch_size=self.vision_batch_size, 
                                       attn_implement=config._attn_implementation,
                                       model_path=self.model_path)
            logger.debug(f"✨building engine Config={cfg}")
            self.model.vision_model_trt = build_engine(
                cfg=cfg,
                model=vision_model,
                device=torch.device("cuda"))
            del self.model.visual
            self.model.visual = vision_model
            os.system(f"rm -rf {cfg['onnx_path']}")

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""
        from qwen_vl_utils import process_vision_info

        images = self.collect_images(messages)
        optional_keys = {'resized_height', 'resized_width', 'min_pixels', 'max_pixels'}
        content = []
        for image, params in images:
            image = image.convert('RGB')
            item = dict(type='image', image=image)
            item.update({key: params[key] for key in params.keys() if key in optional_keys})
            content.append(item)
        
        image_inputs, _ = process_vision_info([dict(content=content)])
        result = self.processor.image_processor(images=image_inputs, videos=None, return_tensors='pt')
        merge_length = self.processor.image_processor.merge_size**2
        image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
        result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=result))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        image_inputs = [x['content'] for x in messages if x['role'] == 'preprocess'][0]
        outputs = []
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]
        image_tokens = image_inputs["image_tokens"]
        if VLM_ENABLE_TRT:
            new_hidden_states, rotary_pos_emb, attention_mask, cu_attention_mask, window_index = \
                self.model.visual.prepare_input(pixel_values, image_grid_thw, self.hf_config._attn_implementation)
            input_data = {
                "pixel_values": new_hidden_states.cuda(),
                "rotary_pos_emb": rotary_pos_emb.cuda(),
                "attention_mask": attention_mask.cuda(),
                "cu_attention_mask": cu_attention_mask.cuda(),
            }
            hidden_states = self.model.vision_model_trt(input_data)
            reverse_indices = torch.argsort(window_index)
            image_embeds = hidden_states[reverse_indices, :]
            image_embeds = image_embeds.to(dtype=torch.float)
        else:
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw).to(dtype=torch.float)
            logger.info(f"vision_encoder: pixel_values: {list(pixel_values.shape)}, image_embeds:{list(image_embeds.shape)}")
            image_embeds = image_embeds.split(image_tokens.tolist())
        for i, embeddings in enumerate(image_embeds):
            outputs.append(
                dict(embeddings=embeddings,
                    grid_thw=image_inputs['image_grid_thw'][i].tolist()))
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
            prompt = content[0]
            if IMAGE_TOKEN in prompt and '<|vision_start|>' not in prompt:
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>')
            else:
                # Qwen2-VL-2B-Instruct will concat image and user prompt
                # according to their order in the content list
                # we insert image token before user prompt by default. The
                # user can use custom image token position if they want the
                # same decorated prompt as Qwen2-VL
                prompt = f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>' * \
                    n_images + prompt
            prompt_messages.append(dict(role=message['role'], content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def _get_mrope_info(self,
                       seq_len: int,
                       grid_thws: List[Tuple[int, int, int]] = None,
                       embedding_ranges: List[Tuple[int, int]] = None):
        if grid_thws is None:
            mrope_position_ids = torch.arange(seq_len).expand(3, -1)
            mrope_position_delta = torch.tensor([0], dtype=torch.long)
        else:
            mrope_position_ids = [
                torch.arange(embedding_ranges[0][0]).expand(3, -1)
            ]
            st_idx = embedding_ranges[0][0]
            for i, (grid_thw, embedding_range) in enumerate(
                    zip(grid_thws, embedding_ranges)):
                llm_grid_t, llm_grid_h, llm_grid_w = grid_thw
                llm_grid_h //= 2
                llm_grid_w //= 2
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()
                mrope_position_ids.append(
                    torch.stack([t_index, h_index, w_index]) + st_idx)
                st_idx += max(llm_grid_h, llm_grid_w)
                if i < len(embedding_ranges) - 1:
                    text_len = embedding_ranges[i +
                                                1][0] - embedding_ranges[i][1]
                else:
                    text_len = seq_len - embedding_range[1]
                mrope_position_ids.append(
                    torch.arange(text_len).expand(3, -1) + st_idx)
                st_idx += text_len
            mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)
            mrope_position_delta = torch.tensor([st_idx - seq_len],
                                                dtype=torch.long)

        return mrope_position_ids, mrope_position_delta
    
    def to_turbomind_aux(self, messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start):
        """Auxiliary function to pack the forwarding results in a format
        compatible with what is required by turbomind engine.

        Args:
            messages(List[Dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        # collect image features from messages
        features = [x['content'] for x in messages if x['role'] == 'forward'][0]
        grid_thws = [x['grid_thw'] for x in features]
        features = [x['embeddings'] for x in features]
        features = [x.cpu().numpy() for x in features]
        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(features) + 1, (f'the number of {IMAGE_TOKEN} is not equal '
                                                f'to input images, {len(segs) - 1} vs {len(features)}')

        # tokenizer prompt, and get input_embeddings and input_embedding_ranges
        input_ids = []
        begins = []
        ends = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(features):
                image_dim = features[i - 1].shape[0]
                begins.append(len(input_ids))
                ends.append(begins[-1] + image_dim)
                input_ids.extend([self.image_token_id] * image_dim)
            seg_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        ranges = np.stack([begins, ends], axis=1).tolist()
        # Qwen2.5VL MRope Position IDs
        mrope_position_ids, mrope_position_delta = self._get_mrope_info(
                                seq_len=len(input_ids), grid_thws=grid_thws, embedding_ranges=ranges)
        return dict(prompt=prompt, 
                    input_ids=input_ids, 
                    input_embeddings=features, 
                    input_embedding_ranges=ranges,
                    mrope_position_ids=mrope_position_ids,
                    mrope_position_delta=mrope_position_delta)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
    
    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        """Return to the information needed by pytorch engine."""
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

def UT_Qwen2_5VL():
    from transformers import AutoConfig
    from lmdeploy.vl.utils import load_image
    model_dir = "/home/wenlong.cao/models/Qwen2.5-VL-7B-Instruct/"
    os.environ["ONELLM_VLM_ENABLE_TRT"] = "1"
    max_bz = 1
    print(f"max_bz={max_bz}")
    print(f"VLM_ENABLE_TRT={VLM_ENABLE_TRT}")
    os.environ["ONELLM_TRT_VISION_MAX_BATCH_SIZE"] = str(max_bz)
    messages = [
        dict(role='user', content=[
            dict(type='text', text="You are an AI assistent. Give me a short description for the image."),
            dict(type='image', image=load_image('https://cf.shopee.sg/file/vn-11134207-7qukw-lgbyq7x8fbav0b').resize((448, 448)),
                cache_hit=False),
        ])
    ]
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen2d5VLModel(model_path=model_dir, with_llm=False, hf_config=hf_config)
    model.build_preprocessor()
    model.build_model()
    messages = model.preprocess(messages)
    messages = model.forward(messages, max_batch_size=max_bz)
    print(f"messages={messages[-1]['content']}")
    
    
if __name__ == "__main__":
    UT_Qwen2_5VL()
# A100 tensorrt 
# export ONELLM_VLM_ENABLE_TRT=1 python3 lmdeploy/vl/model/qwen2_5.py
# 2025-04-23 09:10:47,573 - lmdeploy - INFO - workflow.py:72 - 🔔[ONELLM] speed up 3.67, 58.187ms vs 15.858ms
# batchsize=1 time: 52.934ms

# A100 
# export ONELLM_ENABLE_FLASH_ATTN=1
# batchsize=1 time: 90.035ms