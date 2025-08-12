import gc
import re
from typing import Optional, Tuple, List
import copy
import math

import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM


# HACK: Override the dtype_byte_size function in transformers to support float8 types
# Fix is posted upstream https://github.com/huggingface/transformers/pull/30488
def new_dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
        )
    else:
        if torch.isnan(tensor).any():
            print(f"Found {torch.isnan(tensor).sum()} NaN in {tensor.numel()} tensor, replacing NaNs with 0")
            tensor = torch.nan_to_num(tensor, nan=0)
        if torch.isinf(tensor).any():
            print(f"Found {torch.isinf(tensor).sum()} Inf in {tensor.numel()} tensor, replacing Infs with minmax")
            max_val_t = tensor[~torch.isinf(tensor)].max() if (tensor == float('inf')).any() else tensor.max()
            min_val_t = tensor[~torch.isinf(tensor)].min() if (tensor == float('-inf')).any() else tensor.min()
            tensor = torch.nan_to_num(tensor, posinf=max_val_t.item(), neginf=min_val_t.item())
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    assert torch.isfinite(scale), "scale must be a finite number"
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale


def static_per_tensor_quantize(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    output = torch.nn.functional.linear(
        A.to(out_dtype) * A_scale,
        B.to(out_dtype) * B_scale.to(out_dtype),
        bias=bias,
    )
    return output


# Class responsible for quantizing weights
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias

    def forward(self, x):
        qinput, x_scale = per_tensor_quantize(x)
        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


# Module responsible for taking already quantized weights, and recording input
# scales (and possibly output scales) using an activation observer
class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        quantize_output: bool = False,
        name: str = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.input_scale = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.output_scale = None
        self.quantize_output = quantize_output
        self.name = name

    def forward(self, x):
        qinput, x_input_scale = per_tensor_quantize(x)
        # check x_input_scale is inf
        if not torch.isfinite(x_input_scale):
            print(f'Found input scale of {self.name} layer is inf/nan under this calibration sample, skip update.')
        elif x_input_scale > self.input_scale:
#            print(f'{self.name}: self.input_scale={self.input_scale.data}, x_input_scale={x_input_scale}')
            self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        # Optionally, quantize output and record scale
        if self.quantize_output:
            qoutput, output_scale = per_tensor_quantize(output)
            if self.output_scale is None:
                self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            elif output_scale > self.output_scale:
                self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            output = qoutput.to(output.dtype) * output_scale

        return output


# Module responsible for representing the final checkpoint representation
class FP8StaticLinear(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.nn.Parameter,
        weight_scale: torch.nn.Parameter,
        bias: torch.nn.Parameter,
        input_scale: torch.nn.Parameter,
        output_scale: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        self.qweight = qweight
        self.weight_scale = weight_scale
        self.bias = bias
        self.input_scale = input_scale
        self.output_scale = output_scale

    def forward(self, x):
        qinput = static_per_tensor_quantize(x, self.input_scale)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.qweight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        if self.output_scale:
            qoutput = static_per_tensor_quantize(output, self.output_scale)
            output = qoutput.to(output.dtype) * self.output_scale

        return output


def replace_module(model: AutoModelForCausalLM,
                   name: str,
                   new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)


def get_kv_cache_quant_layers(model: AutoModelForCausalLM,
                              kv_cache_targets: List[str] = ["k_proj", "v_proj"]
                              ) -> List[str]:
    kv_cache_quant_layers = []

    for name, linear in model.named_modules():
        if not isinstance(linear, torch.nn.Linear):
            continue

        for quant_target in kv_cache_targets:
            if name.endswith(quant_target):
                kv_cache_quant_layers.append(name)

    return kv_cache_quant_layers


def find_matches(value: str, ignored_layers: List[str]) -> bool:
    # returns True if any name in the ignored layers list that matchs the value
    for target in ignored_layers:
        if target.startswith("re:"):
            pattern = target[3:]
            if re.match(pattern, value):
                return True
        elif target == value:
            return True


def quantize_weights(
    model: AutoModelForCausalLM,
    ignored_layers: List[str] = ['lm_head'],
):
    named_modules = list(model.named_modules())
    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if (not isinstance(linear, torch.nn.Linear) or find_matches(name, ignored_layers)):
            continue

        quant_weight, weight_scale = per_tensor_quantize(linear.weight)
        bias = copy.deepcopy(linear.bias) if linear.bias is not None else None
        quant_linear = FP8DynamicLinear(
            weight=quant_weight, weight_scale=weight_scale, bias=bias
        )
        replace_module(model, name, quant_linear)
        del linear.weight
        del linear.bias
        del linear
    cleanup_memory()


def quantize_activations(
    model: AutoModelForCausalLM,
    calibration_tokens: torch.Tensor,
    kv_cache_quant_layers: List[str],
    ignored_layers: List[str] = ['lm_head'],
):
    # Replace weight quantizer with a dynamic activation quantizer observer
    for name, dynamic_quant_linear in model.named_modules():
        if (
            not isinstance(dynamic_quant_linear, FP8DynamicLinear)
            or find_matches(name, ignored_layers)
        ):
            continue
        quantizer = FP8StaticLinearQuantizer(
            weight=dynamic_quant_linear.weight,
            weight_scale=dynamic_quant_linear.weight_scale,
            bias=dynamic_quant_linear.bias,
            quantize_output=(name in kv_cache_quant_layers),
            name = name,
        )
        replace_module(model, name, quantizer)
        del dynamic_quant_linear
    cleanup_memory()

    # Pass through calibration data to measure activation scales
    with torch.inference_mode():
        with tqdm.tqdm(total=calibration_tokens.shape[0],
                       desc="Calibrating activation scales") as pbar:
            for row_idx in range(calibration_tokens.shape[0]):
                model(calibration_tokens[row_idx].reshape(1, -1))
                cleanup_memory()
                pbar.update(1)

    # Replace dynamic quantizer observer with StaticLinear for export
    for name, quantizer in model.named_modules():
        if (
            not isinstance(quantizer, FP8StaticLinearQuantizer)
            or name in ignored_layers
        ):
            continue
        static_proj = FP8StaticLinear(
            qweight=quantizer.weight,
            weight_scale=quantizer.weight_scale,
            bias=quantizer.bias,
            input_scale=quantizer.input_scale,
            output_scale=quantizer.output_scale,
        )
        replace_module(model, name, static_proj)
        del quantizer
    cleanup_memory()

    # Post-process step for kv cache scales to take the k/v module
    # `output_scale` parameters, and store them in the parent attention
    # module as `k_scale` and `v_scale`
    if kv_cache_quant_layers:
        # Assumes that list is ordered such that [layer0.k_proj, layer0.v_proj, layer1.k_proj, layer1.v_proj, ...]
        # so we make a list of tuples [(layer0.k_proj, layer0.v_proj), (layer1.k_proj, layer1.v_proj), ...]
        kv_proj_pairs = zip(*[iter(kv_cache_quant_layers)]*2)
        for k_proj_name, v_proj_name in kv_proj_pairs:
            parent_module_name = ".".join(k_proj_name.split(".")[:-1])
            assert parent_module_name == ".".join(v_proj_name.split(".")[:-1])
            parent_module = dict(model.named_modules())[parent_module_name]

            k_proj = dict(model.named_modules())[k_proj_name]
            v_proj = dict(model.named_modules())[v_proj_name]

            parent_module.k_scale = torch.nn.Parameter(
                k_proj.output_scale, requires_grad=False
            )
            parent_module.v_scale = torch.nn.Parameter(
                v_proj.output_scale, requires_grad=False
            )

            # Remove output_scale from k_proj and v_proj
            k_proj.output_scale = None
            v_proj.output_scale = None

    cleanup_memory()
