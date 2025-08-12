import os.path as osp
import math
import json

from .base import INPUT_MODELS
from ..config import RopeParam
from .llama import LlamaModel, LlamaReader

class CompassReader(LlamaReader):
    """CompassLLVMReader for llama model."""

    attn_layer_prefix = 'llm.model.layers'
    attn_layer_patten = r'llm.model.layers.([0-9]+).'
    tok_embeddings_key = 'llm.model.embed_tokens.weight'
    norm_weight_key = 'llm.model.norm.weight'
    output_weight_key = 'llm.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict, policy):
        model_cfg = model_cfg.get('llm_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, policy)


class CompassV1d6Reader(CompassReader):
    """CompassLLVMReader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

class CompassMoeReader(CompassV1d6Reader):
    ffn_pattern = r'shared_expert\.'

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts')
        result = []
        key_list = []
        if self.model_cfg['num_experts'] == 16:
            # compass-max
            key_list = ['w1', 'w2', 'w3']
            name = f'language_model.model.layers.{i}.block_sparse_moe.experts.{e}.KEY.{kind}'
        else:
            # compass-smoe
            key_list = ['gate', 'down', 'up']
            name = f'language_model.model.layers.{i}.mlp.experts.{e}.KEY_proj.{kind}'

        for key in key_list:
            _name = name.replace("KEY", key)
            tensor = self.params.get(_name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i):
        if self.model_cfg['num_experts'] == 16:
            # compass-max
            return self.params.get(f'language_model.model.layers.{i}.block_sparse_moe.gate.weight')
        else:
            return self.params.get(f'language_model.model.layers.{i}.mlp.gate.weight')

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        if not kind:
            return self.filter(self.ffn_pattern)
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[
                f'language_model.model.layers.{i}.mlp.shared_expert.{key}_proj.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_shared_gate(self, i):
        return self.params.get(
            f'language_model.model.layers.{i}.mlp.shared_expert_gate.weight')

@INPUT_MODELS.register_module(name='compassllvm')
class CompassLLVM(LlamaModel):
    """InternVL model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, ckpt_path: str = None, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        self.model_path = model_path
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.version = getattr(config, "version", "1.0")
        # print(f"CompassLLVM version={self.version}")
        version_readers = {
            "1.0": CompassReader,
            "1.6": CompassV1d6Reader,
            "2.0": CompassMoeReader,
        }
        self.Reader = version_readers[self.version]

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)['llm_config']
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
                else:
                    raise RuntimeError(f'Unsupported rope type: {scaling_type}')

            # get tie_word_embeddings, use_normhead
            tie_word_embeddings = model_arg.get('tie_word_embeddings', False)
            use_normhead = model_arg.get('use_normhead', False)
            use_logn_attn = model_arg.get('use_logn', True)
        
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
        if self.version == "2.0":
            # compass-moe
            expert_num = model_arg['num_experts']
            expert_inter_size = model_arg['moe_intermediate_size']
            experts_per_token = model_arg['num_experts_per_tok']
            inter_size = model_arg['shared_expert_intermediate_size']
            moe_shared_gate = model_arg['use_shared_expert_gate']
            norm_topk_prob = model_arg['norm_topk_prob']
            info.update(expert_num=expert_num,
                        expert_inter_size=expert_inter_size,
                        experts_per_token=experts_per_token,
                        inter_size=inter_size,
                        moe_shared_gate=moe_shared_gate,
                        norm_topk_prob=norm_topk_prob,
                        attn_bias=0)

        return info