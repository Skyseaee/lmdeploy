#-*- encoding: utf-8 -*-
"""
Support Qwen2.5-VL-7B-Instruct model using turbomind
Author: wenlong.cao@shopee.com
Date: 2025-02-24 17:00:12
Update: 2025-07-25 10:00:00 refactor for v0.9.1 and test MultiVisionEncoder
"""
import sys
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import transformers

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.onepiece.utils import device_default_half_type

logger = get_logger('lmdeploy')


def check_qwen3_vl_deps_install():
    """check qwen_vl_utils."""
    if transformers.__version__ < '4.57.0':
        raise ImportError('please install transformers>= 4.57.0')
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


@VISION_MODELS.register_module()
class Qwen3VLModel(VisonModel):
    """Qwen3VL model."""

    _arch = 'Qwen3VLMoeForConditionalGeneration'
    
    def build_preprocessor(self):
        check_qwen3_vl_deps_install()
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(image_token)[-1]

    def build_model(self):
        check_qwen3_vl_deps_install()
        from transformers import Qwen3VLMoeForConditionalGeneration
        if self.backend == "pytorch":
            self.vl_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(self.model_path, device_map='cpu')
            return

        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        with init_empty_weights():
            config = self.hf_config
            model = Qwen3VLMoeForConditionalGeneration._from_config(config, 
                                                                    torch_dtype=config.torch_dtype, 
                                                                    attn_implementation=config._attn_implementation) 
        if not self.with_llm:
            del model.model.language_model
            del model.lm_head
        else:
            self.vl_model = model

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map={"":"cuda:0"} if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=[],
                dtype=self.hf_config.torch_dtype)

        self.model = model.eval()
        self.model.visual = self.model.model.visual
        logger.info(f"🔔 model_dtype={self.model.dtype}, attn_implementation={config._attn_implementation}")


    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""
        from qwen_vl_utils import process_vision_info

        images = self.collect_images(messages)
        optional_keys = {'resized_height', 'resized_width', 'min_pixels', 'max_pixels'}
        if self.backend == "pytorch":
            outputs = []
            for image, params in images:
                image = image.convert('RGB')
                item = dict(type='image', image=image)
                item.update({key: params[key] for key in params.keys() if key in optional_keys})
                image_inputs, _ = process_vision_info([dict(content=[item])])
                result = self.processor.image_processor(images=image_inputs, videos=None, return_tensors='pt')
                merge_length = self.processor.image_processor.merge_size**2
                image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
                result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
                outputs.append(result)
            messages.append(dict(role='preprocess', content=outputs))
        else:
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
        pixel_values = image_inputs["pixel_values"].to(self.model.visual.device)
        image_grid_thw = image_inputs["image_grid_thw"].to(self.model.visual.device)
        image_tokens = image_inputs["image_tokens"]
        
        image_embeds, deepstack_feats = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        logger.info(f"vision_encoder: pixel_values: {list(pixel_values.shape)}, image_embeds:{list(image_embeds.shape)}, deepstack_feats: {len(deepstack_feats)}x{deepstack_feats[0].shape}")
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