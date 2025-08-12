# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import torch

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.builder import load_vl_model


logger = get_logger('lmdeploy')


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    """Raise exception on finish."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        raise e

def _default_device(tp, num_encoders, i):
    if tp == 1 or num_encoders == 1:
        logger.info(f"vision_encoder[{i}]: auto")
        return 'auto'
    else:
        step = max(tp // num_encoders, 1)
        gpu_id = (i*step)% tp
        logger.info(f"vision_encoder[{i}]: cuda:{gpu_id}")
        return {'':f"cuda:{gpu_id}"}

class ImageEncoder:
    """Image encoder."""

    def __init__(
        self,
        model_path: str,
        backend: str,
        vision_config: VisionConfig = None,
        backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
    ):
        if vision_config is None:
            vision_config = VisionConfig()
        self.vision_config = vision_config
        self.max_batch_size = vision_config.max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=self.vision_config.instance_num)

        self.model_queue = asyncio.Queue(maxsize=self.vision_config.instance_num)
        for i in range(self.vision_config.instance_num):
            model = load_vl_model(model_path, backend, 
                                backend_config=backend_config,
                                default_device=_default_device(backend_config.tp, vision_config.instance_num, i))
            self.model_queue.put_nowait(model)
        torch.cuda.empty_cache()

    async def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Preprocess multimodal data in the messages."""
        model = await self.model_queue.get()
        future = asyncio.get_event_loop().run_in_executor(self.executor, model.preprocess, messages)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        await self.model_queue.put(model)
        return outputs

    async def async_infer(self, messages: List[Dict]) -> List[Dict]:
        """Get multimodal embedding.

        Args:
            messages (List[Dict]): a list of message, which is the output
            of `preprocess()`
        """
        model = await self.model_queue.get()
        future = asyncio.get_event_loop().run_in_executor(self.executor, model.forward, messages,
                                                          self.max_batch_size)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        await self.model_queue.put(model)
        return outputs

    async def wrap_for_pytorch(self, messages: List[Dict], chat_template, tokenizer, sequence_start) -> List[Dict]:
        """
        Args:
            messages (List[Dict]): a list of message, which is supposed to be
                the output of `preprocess`
        Returns:
            a dict which will be passed to pytorch engine_instance's forward.
            The dict is like the following:
            Dict(
                'prompt': 'the prompt after applying chat template'
                'input_ids': [],
                'multimodal': {
                    'pixel_values': torch.Tensor,
                    ...
                ]
            )
        """
        model = await self.model_queue.get()
        result = model.to_pytorch(messages, chat_template, tokenizer, sequence_start)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
        await self.model_queue.put(model)
        return result

    async def wrap_for_turbomind(self, messages: List[Dict], chat_template, tokenizer, sequence_start) -> Dict:
        """
        Args:
            messages (List[Dict]): a list of message, which is supposed to be
                the output of `async_infer`
        Returns:
            a dict which will be passed to pytorch engine_instance's forward.
            The dict is like the following:
            Dict(
                'prompt': 'the prompt after applying chat template'
                'input_ids': [],
                'input_embeddings': list[torch.Tensor],
                'input_embedding_ranges': list[torch.Tensor],
                ...
        """
        model = await self.model_queue.get()
        result = model.to_turbomind(messages, chat_template, tokenizer, sequence_start)
        # clear data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
                messages[i]['forward'] = None
        await self.model_queue.put(model)
        return result
