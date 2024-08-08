import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity

#Compass-13B Model Param
HIDDEN_SIZES = [5120, 13824]
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096]
INTER_SIZE = [5120, 13824, 15360, 27648]
SEEDS = [0]

def compute_result(output, output_ref):
    return torch.max(torch.abs(output - output_ref)), torch.mean(torch.abs(output - output_ref)), \
        torch.mean(torch.abs(output - output_ref)) / torch.mean(torch.abs(output_ref))

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("inter_size", INTER_SIZE)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant_vs_fp16(num_tokens: int, hidden_size: int, inter_size: int,
                                       seed: int) -> None:
    if hidden_size==13824 and inter_size ==13824: return
    if hidden_size==13824 and inter_size ==15360: return
    if hidden_size==13824 and inter_size ==27648: return

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # create test inputs
    # Note: cuBLASLt float8 matmul requires column major
    #        for the second argument
    x = torch.randn (num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    w = torch.randn (inter_size, hidden_size, dtype=torch.float16, device='cuda').t()

    # do a scaled cast to float8 on the inputs
    x_f8, x_inv_s = to_float8(x)
    w_f8, w_inv_s = to_float8(w)

    fp16_res = torch.mm(x, w)
    # perform the float8 matmul
    fp8_res, _ = torch._scaled_mm(x_f8, w_f8, out_dtype=torch.float16,
                             scale_a=x_inv_s , scale_b=w_inv_s)


    # compare output of float8 matmul to the fp16 baseline
    #cos_sim = F.cosine_similarity(fp16_res.reshape(-1), fp8_res.reshape(-1), dim=0)
    cos_sim=cosine_similarity(fp16_res.reshape(-1).reshape(1,-1).cpu(), fp8_res.reshape(-1).reshape(1,-1).cpu())
    # Cosine similarity between scaled mm and reference
    # should be close to 1.0

    max_diff, mean_diff, rel_diff = compute_result(fp8_res, fp16_res)
    print(f"[{num_tokens},{hidden_size}] x [{hidden_size},{inter_size}] {max_diff} {mean_diff} {rel_diff} {cos_sim[0][0]}")