/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/turbomind/core/quant_mode.h"
#include "src/turbomind/kernels/cutlass_kernels/include/moe_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace tensorrt_llm::plugins
{

std::uintptr_t constexpr kCudaMemAlign = 128;

struct GemmIDMoe
{
    int gemm_idx;
    int num_experts{};
    int moe_k{};
    tensorrt_llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};
    int64_t hidden{};
    int64_t inter{};
    tensorrt_llm::kernels::cutlass_kernels::ActivationType actfn{};
    turbomind::DataType dtype{};
    turbomind::DataType wdtype{};
    turbomind::QuantMode quant_mode;
    bool determinism_mode = false;

    bool operator==(GemmIDMoe const& id) const
    {
        return id.gemm_idx == gemm_idx && id.num_experts == num_experts && id.moe_k == moe_k
            && id.parallelism_config == parallelism_config && id.hidden == hidden && id.inter == inter
            && id.actfn == actfn && id.dtype == dtype && id.wdtype == wdtype && id.quant_mode == quant_mode
            && id.determinism_mode == determinism_mode;
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIDMoe const& id)
    {
        out << "gemm idx, experts, k, parallelism_config, hidden, inter, actfn, dtype, weight "
               "type, parallelism mode, determinism mode="
            << id.gemm_idx << "," << id.num_experts << "," << id.moe_k << "," << id.parallelism_config << ","
            << id.hidden << "," << id.inter << "," << static_cast<int>(id.actfn) << "," << static_cast<int>(id.dtype)
            << "," << static_cast<int>(id.wdtype) << "," << id.quant_mode.value() << "," << id.determinism_mode;
        return out;
    }
};

// Hash of GemmIDMoe
struct GemmIDMoeHash
{
    std::size_t operator()(GemmIDMoe const& id) const
    {
        size_t hash = std::hash<int>{}(id.gemm_idx);
        hash ^= std::hash<int>{}(id.num_experts);
        hash ^= std::hash<int>{}(id.moe_k);
        hash ^= std::hash<int>{}(id.parallelism_config.tp_size);
        hash ^= std::hash<int>{}(id.parallelism_config.ep_size);
        hash ^= std::hash<int>{}(id.parallelism_config.tp_rank);
        hash ^= std::hash<int>{}(id.parallelism_config.ep_rank);
        hash ^= std::hash<int>{}(id.hidden);
        hash ^= std::hash<int>{}(id.inter);
        hash ^= std::hash<int>{}(static_cast<int>(id.actfn));
        hash ^= std::hash<int>{}(static_cast<int>(id.dtype));
        hash ^= std::hash<int>{}(static_cast<int>(id.wdtype));
        hash ^= std::hash<int>{}(static_cast<int>(id.quant_mode.value()));
        return hash;
    }
};

inline int8_t* nextWorkspacePtr(
    int8_t* const base, uintptr_t& offset, uintptr_t const size, uintptr_t const alignment = kCudaMemAlign)
{
    uintptr_t curr_offset = offset;
    uintptr_t next_offset = curr_offset + ((size + alignment - 1) / alignment) * alignment;
    int8_t* newptr = size == 0 ? nullptr : base + curr_offset;
    offset = next_offset;
    return newptr;
}

inline size_t calculateTotalWorkspaceSize(
    size_t const* workspaces, int count, uintptr_t const alignment = kCudaMemAlign)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % alignment)
        {
            total += alignment - (workspaces[i] % alignment);
        }
    }
    return total;
}

// Write values into buffer
template <typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

} // namespace tensorrt_llm::plugins::utils
