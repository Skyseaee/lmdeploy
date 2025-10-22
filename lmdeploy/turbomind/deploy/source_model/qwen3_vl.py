import os.path as osp
import math
import json

import torch
from typing import Tuple
from .base import INPUT_MODELS
from ..config import RopeParam

from .qwen import Qwen3MoeReader, Qwen3MoeModel

class Qwen3VLMoEReader(Qwen3MoeReader):
    """Qwen3VLMoEReader for qwen3vl-moe model.
    """
    attn_layer_prefix  = 'model.language_model.layers'
    attn_layer_patten = r'model.language_model.layers.([0-9]+).'
    tok_embeddings_key = 'model.language_model.embed_tokens.weight'
    norm_weight_key    = 'model.language_model.norm.weight'
    output_weight_key  = 'lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg['text_config']
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)
    
    def qk_norm(self, i: int):
        result = []
        for x in ['q', 'k']:
            name = f'{self.attn_layer_prefix}.{i}.self_attn.{x}_norm.weight'
            result.append(self.transform(self.params.get(name), 'weight'))
        return (*result, )
    
    def merged_moe_ffn_expert(self, e=None, i=None, expert_num=None):
        if not expert_num:
            return self.filter(r'experts')
        def split_gate_up_weight(t: torch.Tensor, E: int) -> Tuple[torch.Tensor, torch.Tensor]:
            # (E, H, 2I)
            assert t.ndim == 3 and t.shape[0] == E
            gate_part, up_part = torch.chunk(t, 2, dim=2)
            return gate_part, up_part
        # ----- gate_up_proj [128, 2048, 1536] [expert_num, hidden_dim, expert_dim*2]
        name = f'model.language_model.layers.{i}.mlp.experts.gate_up_proj'
        gate_up_tensor = self.params.get(name)
        # gate [128, 2048, 768]
        # up   [128, 2048, 768]
        gate_proj, up_proj = split_gate_up_weight(gate_up_tensor, expert_num)
        
        # ----- down_proj [128, 768, 2048] [expert_num, expert_dim, hidden_dim]
        name = f'model.language_model.layers.{i}.mlp.experts.down_proj'
        down_tensor = self.params.get(name) 
        down_proj = down_tensor[e]
        result = [gate_proj[e].contiguous(), down_proj, up_proj[e].contiguous()]
        return (*result, )

    def moe_ffn_gate(self, i):
        t = self.transform(self.params.get(f'model.language_model.layers.{i}.mlp.gate.weight'), 'weight')
        return t


@INPUT_MODELS.register_module(name='qwen3_vl_moe')
class Qwen3VLMoE(Qwen3MoeModel):
    """Qwen3VLMoE model in hf format."""
    Reader = Qwen3VLMoEReader
    
    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)['text_config']
            print(model_arg)
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            attn_head_num = model_arg['num_attention_heads']
            vocab_size = model_arg['vocab_size']
            inter_size = model_arg['intermediate_size']
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
            hidden_units = model_arg['hidden_size']
            head_dim = model_arg.get('head_dim', hidden_units // attn_head_num)
            # compute rope param
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(model_arg.get('max_position_embeddings', 0))
            rope_param = RopeParam(type='default', base=rope_theta, dim=head_dim)
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            use_dynamic_ntk = 0
            scaling_type = ''
            low_freq_factor = 1.0
            high_freq_factor = 1.0
            attention_factor = -1.0
            beta_fast = 32.0
            beta_slow = 1.0
            mrope_section = None
            original_max_position_embeddings = 0
            if isinstance(rope_scaling, dict):
                llama2_scaling_type = rope_scaling.get('type', '')
                llama3_scaling_type = rope_scaling.get('rope_type', '')
                if llama2_scaling_type and llama3_scaling_type \
                        and llama2_scaling_type != llama3_scaling_type:
                    raise ValueError(f'Ambiguous rope_scaling in config: {model_arg}')
                scaling_type = llama2_scaling_type if llama2_scaling_type \
                    else llama3_scaling_type
                scaling_factor = rope_scaling.get('factor', 0.0)
                if scaling_type == 'dynamic':
                    rope_param.__dict__.update(type='dynamic',
                                               factor=scaling_factor,
                                               max_position_embeddings=max_position_embeddings)
                elif scaling_type == 'linear':
                    rope_param.__dict__.update(type='linear', factor=scaling_factor)
                elif scaling_type == 'llama3':
                    low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
                    high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
                    original_max_position_embeddings = model_arg['rope_scaling'].get(
                        'original_max_position_embeddings', 0)
                    rope_param.__dict__.update(type='llama3',
                                               factor=scaling_factor,
                                               low_freq_factor=low_freq_factor,
                                               high_freq_factor=high_freq_factor,
                                               original_max_position_embeddings=original_max_position_embeddings)
                elif scaling_type == 'yarn':
                    attention_factor = rope_scaling.get('attention_factor', None)
                    if attention_factor is None:
                        attention_factor = 0.1 * math.log(scaling_factor) + 1.0
                    beta_fast = rope_scaling.get('beta_fast', 32.0)
                    beta_slow = rope_scaling.get('beta_slow', 1.0)
                    rope_param.__dict__.update(type='yarn',
                                               factor=scaling_factor,
                                               max_position_embeddings=max_position_embeddings,
                                               attention_factor=attention_factor,
                                               beta_fast=beta_fast,
                                               beta_slow=beta_slow)
                elif scaling_type == 'mrope':
                    mrope_section = rope_scaling.get('mrope_section', [16, 24, 24])
                elif scaling_type == 'default':
                    pass
                else:
                    raise RuntimeError(f'Unsupported rope type: {scaling_type}')

            # get tie_word_embeddings, use_normhead
            tie_word_embeddings = model_arg.get('tie_word_embeddings', 0)
            use_normhead = model_arg.get('use_normhead', 0)
            use_logn_attn = model_arg.get('use_logn_attn', 0)

            info = dict(
                size_per_head=head_dim,
                rotary_embedding=hidden_units // attn_head_num,
                num_layer=num_layer,
                norm_eps=norm_eps,
                head_num=attn_head_num,
                kv_head_num=kv_head_num,
                hidden_units=hidden_units,
                inter_size=inter_size,
                vocab_size=vocab_size,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                original_max_position_embeddings=original_max_position_embeddings,
                use_dynamic_ntk=use_dynamic_ntk,
                rope_scaling_type=scaling_type,
                rope_scaling_factor=scaling_factor,
                mrope_section=mrope_section,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                attention_factor=attention_factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                tie_word_embeddings=tie_word_embeddings,
                use_normhead=use_normhead,
                use_logn_attn=use_logn_attn,
                rope_param=rope_param)

            info.update(
                qk_norm=True,
                expert_num=model_arg.get('num_experts', 128),
                experts_per_token=model_arg.get('num_experts_per_tok', 8),
                expert_inter_size=model_arg.get('moe_intermediate_size', 768),
                attn_bias=model_arg.get('attention_bias', 0),
                inter_size=0,  # no shared expert
                norm_topk_prob=model_arg.get('norm_topk_prob', False))
            return info