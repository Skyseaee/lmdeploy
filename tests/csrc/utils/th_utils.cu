/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "th_utils.h"

namespace ft = turbomind;

namespace torch_ext {

std::vector<size_t> convert_shape(torch::Tensor tensor)
{
    std::vector<size_t> v_shape;
    for (int i = 0; i < tensor.dim(); i++) {
        v_shape.push_back(tensor.size(i));
    }
    return v_shape;
}

template<typename T>
ft::Tensor convert_tensor(torch::Tensor tensor)
{
    ft::MemoryType mtype = tensor.is_cuda() ? ft::MEMORY_GPU : ft::MEMORY_CPU;
    return convert_tensor<T>(tensor, mtype);
}

template ft::Tensor convert_tensor<int8_t>(torch::Tensor tensor);
template ft::Tensor convert_tensor<float>(torch::Tensor tensor);
template ft::Tensor convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template ft::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template ft::Tensor convert_tensor<int>(torch::Tensor tensor);
template ft::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor);
template ft::Tensor convert_tensor<unsigned int>(torch::Tensor tensor);
template ft::Tensor convert_tensor<bool>(torch::Tensor tensor);

template<typename T>
ft::Tensor convert_tensor(torch::Tensor tensor, ft::MemoryType memory_type)
{
    return ft::Tensor{memory_type, ft::getTensorType<T>(), convert_shape(tensor), get_ptr<T>(tensor)};
}

template ft::Tensor convert_tensor<int8_t>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<float>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<half>(torch::Tensor tensor, ft::MemoryType memory_type);
#ifdef ENABLE_BF16
template ft::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor, ft::MemoryType memory_type);
#endif
template ft::Tensor convert_tensor<int>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<unsigned int>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<bool>(torch::Tensor tensor, ft::MemoryType memory_type);

size_t sizeBytes(torch::Tensor tensor)
{
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}


// 将数据转换为 float32 的帮助函数
torch::Tensor as_float32_tensor(const torch::Tensor& x) {
    return x.to(torch::kFloat32).cuda();
}

// FP8 定义，用于动态每张量的 FP8 量化
std::tuple<torch::Tensor, torch::Tensor> ref_dynamic_per_tensor_fp8_quant(const torch::Tensor& x) {
    auto fp8_traits = torch::finfo(torch::kFloat8_e4m3fn);
    auto fp8_max = as_float32_tensor(fp8_traits.max());
    auto one = as_float32_tensor(1.0f);

    // 计算 x 的绝对值最大值
    auto x_max = as_float32_tensor(x.abs().max());

    // 计算缩放因子
    auto ref_scale = x_max / fp8_max;
    auto ref_iscale = one / ref_scale;

    // 将数据转换并限制在 FP8 范围内，然后转换为 FP8 类型
    auto ref_out = (as_float32_tensor(x) * ref_iscale)
        .clamp(fp8_traits.min(), fp8_traits.max())
        .to(torch::kFloat8_e4m3fn);

    return std::make_tuple(ref_out, ref_scale);
}

}  // namespace torch_ext
