# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

import torch


def ensure_fp8(tensors: torch.Tensor):
    """Ensure tensors in fp8_e4m3fn format."""
    result = []
    for tensor in tensors:
        if tensor is not None:
            assert tensor.dtype == torch.uint8
            result.append(tensor)
        else:
            result.append(None)
    return (*result, )


def ensure_fp32(tensors: torch.Tensor):
    """Ensure tensors in fp32 format."""
    result = []
    for tensor in tensors:
        if tensor is not None:
            assert tensor.dtype == torch.float32
            result.append(tensor)
        else:
            result.append(None)
    return (*result, )


def requantize_qkv(weights: List[torch.Tensor],
                   scales: List[torch.Tensor]):
    # Credit to: https://github.com/vllm-project/vllm/pull/4332#issuecomment-2085560821
    device_q, device_k, device_v = [tensor.device for tensor in weights]
    wq, wk, wv = [tensor.view(dtype=torch.float8_e4m3fn).cpu() if tensor.is_cuda else tensor for tensor in weights]
    wq_scale, wk_scale, wv_scale = [s.cpu() if s.is_cuda else s for s in scales]

    w_scale = max(wq_scale, wk_scale, wv_scale)
    qw = ((wq_scale / w_scale) * wq).view(dtype=torch.uint8).to(device_q)
    kw = ((wk_scale / w_scale) * wk).view(dtype=torch.uint8).to(device_k)
    vw = ((wv_scale / w_scale) * wv).view(dtype=torch.uint8).to(device_v)
    return qw, kw, vw, w_scale.to(device_q)


def fused_w1w3(w1: torch.Tensor, w3: torch.Tensor, tp: int, dim: int):

    def reshape(x):
        return x.view(x.size(0), tp, -1) if dim == 2 else x.view(tp, -1)

    device = w1.device
    if w1.is_cuda and w1.dtype == torch.uint8:
        w1 = w1.view(dtype=torch.float8_e4m3fn).cpu()
    if w3.is_cuda and w3.dtype == torch.uint8:
        w3 = w3.view(dtype=torch.float8_e4m3fn).cpu()

    # w1 means gate, w3 means up
    w31 = torch.cat((reshape(w3), reshape(w1)), dim=0)

    if w31.is_cpu:
        w31 = w31.view(dtype=torch.uint8).to(device)

    # (2 * inter_size, hidden_dim)
    return w31.view(-1, w1.size(1))


def identity(x):
    return x

def inv(x):
    return torch.reciprocal(x)

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

            qkv_weight_scale_inv = torch.reciprocal(qkv_s)
            qkv_input_scale_inv = torch.reciprocal(q_is)
            
            o_weight_scale_inv = torch.reciprocal(o_s)
            o_input_scale_inv = torch.reciprocal(o_is)

            # '.qweight', '.weight_scale', '.input_scale', 
            # '.input_scale_inv', '.weight_scale_inv', '.host_input_scale_inv'
            # qweight for qkv and o
            f(i, qkvo, 'qweight', identity)
            # scales for qkv
            f(i, qkv_s.unsqueeze(0), 'w_qkv.weight_scale', to_float)
            f(i, q_is.unsqueeze(0), 'w_qkv.input_scale', to_float)
            f(i, qkv_input_scale_inv.unsqueeze(0), 'w_qkv.input_scale_inv', to_float)
            f(i, qkv_input_scale_inv.unsqueeze(0), 'w_qkv.host_input_scale_inv', to_float)
            f(i, qkv_weight_scale_inv.unsqueeze(0), 'w_qkv.weight_scale_inv', to_float)
            
            # scales for o
            f(i, o_s.unsqueeze(0), 'wo.weight_scale', to_float)
            f(i, o_is.unsqueeze(0), 'wo.input_scale', to_float)
            f(i, o_input_scale_inv.unsqueeze(0), 'wo.input_scale_inv', to_float)
            f(i, o_input_scale_inv.unsqueeze(0), 'wo.host_input_scale_inv', to_float)
            f(i, o_weight_scale_inv.unsqueeze(0), 'wo.weight_scale_inv', to_float)
        elif module_type == 'ffn':
            # qweight, weight_scale and input_scale for w1, w2 and w3
            f(i, ensure_fp8(g('qweight')), 'qweight', identity)
            f(i, ensure_fp32(g('weight_scale')), 'weight_scale', to_float)
            f(i, ensure_fp32(g('input_scale')), 'input_scale', to_float)
            f(i, ensure_fp32(g('input_scale')), 'input_scale_inv', inv)
            f(i, ensure_fp32(g('input_scale')), 'host_input_scale_inv', inv)
            f(i, ensure_fp32(g('weight_scale')), 'weight_scale_inv', inv)
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


def get_params(keys: List[str], bias=0, model_format=None, quant_algo=None):
    ps = []
    if PLora.take(keys):
        ps.append(PLora())
    if model_format == 'fp8' and quant_algo == 'fp8_static':
        if QuantWeightFP8.take(keys):
            ps.append(QuantWeightFP8())
    else:
        if QuantWeightOnly.take(keys):
            ps.append(QuantWeightOnly())
    if WeightScaleInv.take(keys):
        ps.append(WeightScaleInv())
    if Weight.take(keys):
        ps.append(Weight())
    if bias and Bias.take(keys):
        ps.append(Bias())
    return ps
