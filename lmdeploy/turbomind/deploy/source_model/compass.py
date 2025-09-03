# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader

class CompassReader(LlamaReader):
    """CompassReader."""

    def output_weight(self):
        """Get output."""
        tensor = self.params.get(self.output_weight_key, None)
        # normhead in Compassllm v0.3
        if tensor is not None and 'use_normhead' in self.model_cfg \
                and self.model_cfg['use_normhead']:
            tensor = torch.nn.functional.normalize(tensor)
        return tensor

@INPUT_MODELS.register_module(name='compass')
class CompassModel(LlamaModel):
    """Compass model in hf format."""

    Reader = CompassReader

class CompassMoeReader(LlamaReader):

    ffn_pattern = r'shared_expert\.'

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts')
        result = []
        key_list = []
        if self.model_cfg['num_experts'] == 16:
            # compass-max
            key_list = ['w1', 'w2', 'w3']
            name = f'model.layers.{i}.block_sparse_moe.experts.{e}.KEY.{kind}'
        else:
            # compass-smoe
            key_list = ['gate', 'down', 'up']
            name = f'model.layers.{i}.mlp.experts.{e}.KEY_proj.{kind}'

        for key in key_list:
            _name = name.replace("KEY", key)
            tensor = self.params.get(_name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i):
        if self.model_cfg['num_experts'] == 16:
            # compass-max
            return self.params.get(f'model.layers.{i}.block_sparse_moe.gate.weight')
        else:
            return self.params.get(f'model.layers.{i}.mlp.gate.weight')

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        if not kind:
            return self.filter(self.ffn_pattern)
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[
                f'model.layers.{i}.mlp.shared_expert.{key}_proj.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_shared_gate(self, i):
        return self.params.get(
            f'model.layers.{i}.mlp.shared_expert_gate.weight')


@INPUT_MODELS.register_module(name='compass-smoe')
@INPUT_MODELS.register_module(name='compass-moe')
class CompassMoeModel(LlamaModel):

    Reader = CompassMoeReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info['expert_num'] = cfg['num_experts']
        info['expert_inter_size'] = cfg['moe_intermediate_size']
        info['experts_per_token'] = cfg['num_experts_per_tok']
        if cfg['num_experts'] == 16:
            # compass-max
            info['inter_size'] = 0
        else:
            # compass-smoe
            info['inter_size'] = cfg['shared_expert_intermediate_size']
        info['moe_shared_gate'] = False
        # hardcoding norm_topk_prob because it's compass-moe/smoe's default behaviour
        info['norm_topk_prob'] = True
        info['attn_bias'] = 0
        return info
