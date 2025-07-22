import os
from glob import glob
import os.path as osp
import json

from lmdeploy.archs import get_model_arch
from .base import INPUT_MODELS
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
        # self.ckpt_path = ckpt_path if ckpt_path else model_path
        # self.ckpt_files = self.get_ckpt()
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.version = "1.0"
        if hasattr(config, "version"):
            self.version = config.version
        version_readers = {
            "1.0": CompassReader,
            "1.6": CompassV1d6Reader,
            "2.0": CompassMoeReader,
        }
        self.Reader = version_readers[self.version]

    def get_ckpt(self):
        """Get weight files."""
        patterns = ['*.safetensors', 'pytorch_model*.bin']
        files = []
        for pattern in patterns:
            files = glob(os.path.join(self.ckpt_path, pattern))
            files = [os.path.basename(file) for file in files]
            if len(files) > 0:
                break
        files = sorted(files)
        return files

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            config = json.load(f)
            model_arg = config['llm_config']
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            attn_head_num = model_arg['num_attention_heads']
            vocab_size = model_arg['vocab_size']
            inter_size = model_arg['intermediate_size']
            hidden_units = model_arg['hidden_size']
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(
                model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            use_logn_attn = model_arg.get('use_logn', True)
            tie_word_embeddings = model_arg.get('tie_word_embeddings', False)
            use_normhead = model_arg.get('use_normhead', False)
            scaling_factor = 0.0
            use_dynamic_ntk = 0
            if isinstance(rope_scaling, dict):
                scaling_type = model_arg['rope_scaling'].get('type', '')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')
                if scaling_type == 'dynamic':
                    use_dynamic_ntk = 1

            info = dict(num_layer=num_layer,
                        norm_eps=norm_eps,
                        attn_head_num=attn_head_num,
                        kv_head_num=kv_head_num,
                        head_num=attn_head_num,
                        hidden_units=hidden_units,
                        rope_theta=rope_theta,
                        vocab_size=vocab_size,
                        inter_size=inter_size,
                        max_position_embeddings=max_position_embeddings,
                        use_dynamic_ntk=use_dynamic_ntk,
                        rope_scaling_factor=scaling_factor,
                        use_logn_attn=use_logn_attn,
                        tie_word_embeddings=tie_word_embeddings,
                        use_normhead=use_normhead)

            if self.version == "2.0":
                # compass-moe
                num_experts = model_arg['num_experts']
                expert_inter_size = model_arg['moe_intermediate_size']
                experts_per_token = model_arg['num_experts_per_tok']
                inter_size = model_arg['shared_expert_intermediate_size']
                moe_shared_gate = model_arg['use_shared_expert_gate']
                moe_norm_topk = model_arg['norm_topk_prob']
                info.update(num_experts=num_experts,
                            expert_inter_size=expert_inter_size,
                            experts_per_token=experts_per_token,
                            inter_size=inter_size,
                            moe_shared_gate=moe_shared_gate,
                            moe_norm_topk=moe_norm_topk,
                            attn_bias=0)
            return info
