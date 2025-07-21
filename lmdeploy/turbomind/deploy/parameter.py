# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

import torch


def identity(x):
    return x


def to_half(x: torch.Tensor):
    return x.to(torch.half)


def to_float(x: torch.Tensor):
    return x.to(torch.float)


def to_fp8(x: torch.Tensor):
    assert x.dtype == torch.uint8
    return x.view(dtype=torch.float8_e4m3fn)


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


class Parameter:
    KEY = ()

    @classmethod
    def take(cls, keys: List[str]):
        if not any(k.endswith(cls.KEYS[0]) for k in keys):
            return False
        xs = []
        for k in keys:
            if any(k.endswith(p) for p in cls.KEYS):
                xs.append(k)
        for x in xs:
            keys.remove(x)
        return True

    @abstractmethod
    def __call__(cls, f, g, i):
        pass


class QuantWeightOnly(Parameter):
    KEYS = '.qweight', '.scales', '.qzeros'

    def __call__(self, f, g, i):
        f(i, g('qweight'), 'qweight', pack_u4_row)
        f(i, g('scales'), 'scales', to_half, apply_gs=True)
        f(i, g('qzeros'), 'zeros', to_half, apply_gs=True)


class QuantWeightFP8(Parameter):
    KEYS = '.qweight', '.weight_scale', '.input_scale'

    def __call__(self, f, g, i, module_type=None):
        if module_type == 'attn':
            # attn, per-tensor, only one scale
            qw, kw, vw, ow = ensure_fp8(g('qweight'))
            q_s, k_s, v_s, o_s = ensure_fp32(g('weight_scale'))
            q_is, k_is, v_is, o_is = ensure_fp32(g('input_scale'))
            assert torch.equal(q_is, k_is) and torch.equal(q_is, v_is), \
                "The input scale for q, k, v are not equal!"

            # requantize the separately quantized q, k, v weights to share with
            # a single weight scale.
            wq, wk, wv, qkv_s = requantize_qkv([qw, kw, vw], [q_s, k_s, v_s])
            qkvo = wq, wk, wv, ow

            # qweight for qkv and o
            f(i, qkvo, 'qweight', identity)
            # scales for qkv
            f(i, qkv_s.unsqueeze(0), 'w_qkv.weight_scale', to_fp32)
            f(i, q_is.unsqueeze(0), 'w_qkv.input_scale', to_fp32)
            # scales for o
            f(i, o_s.unsqueeze(0), 'wo.weight_scale', to_fp32)
            f(i, o_is.unsqueeze(0), 'wo.input_scale', to_fp32)
        elif module_type == 'ffn':
            # qweight, weight_scale and input_scale for w1, w2 and w3
            f(i, ensure_fp8(g('qweight')), 'qweight', identity)
            f(i, ensure_fp32(g('weight_scale')), 'weight_scale', to_fp32)
            f(i, ensure_fp32(g('input_scale')), 'input_scale', to_fp32)
        else:
            raise ValueError(f"Module type {module_type} is not support!")


class WeightScaleInv(Parameter):
    KEYS = '.weight_scale_inv', '.weight'

    # TODO: flag any operations crossing the quant blocks as illegal
    def __call__(self, f, g, i):
        f(i, g('weight_scale_inv'), 'scales', to_float, block_size=128)
        f(i, g('weight'), 'weight', identity)


class Weight(Parameter):
    KEYS = '.weight',

    def __call__(self, f, g, i):
        f(i, g('weight'), 'weight', identity)


class Bias(Parameter):
    KEYS = '.bias',

    def __call__(self, f, g, i):
        f(i, g('bias'), 'bias', identity)


class PLora(Parameter):
    KEYS = '.Plora_A.weight', '.Plora_B.weight'

    def __call__(self, f, g, i):
        f(i, g('Plora_A.weight'), 'lora_a.weight', identity)
        f(i, g('Plora_B.weight'), 'lora_b.weight', identity)


def get_params(keys: List[str], bias=0):
    ps = []
    if PLora.take(keys):
        ps.append(PLora())
    if QuantWeightOnly.take(keys):
        ps.append(QuantWeightOnly())
    if WeightScaleInv.take(keys):
        ps.append(WeightScaleInv())
    if Weight.take(keys):
        ps.append(Weight())
    if bias and Bias.take(keys):
        ps.append(Bias())
    return ps
