/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

namespace turbomind {

// Note(meng): Compute moe select experts and final scales
//template<typename T>
void invokeMoESelectExpertAndFinalScales(float const*  gating_output,           // gate result
                                         float*        token_final_scales,      // final scales(softmax+renorm(optional)) results
                                         int*          token_selected_experts,  // select experts index
                                         float*        softmax_out,             // if (!is_pow_2 || num_experts > 256), we need softmax_out to save res
                                         int64_t const num_rows,                // tokens num
                                         int const     num_experts,             // experts num
                                         int const     k,                       // expert per token
                                         int const     start_expert,            // start expert in ep
                                         int const     end_expert,              // end expert in ep
                                         cudaStream_t  stream);                 // stream
}  // namespace turbomind
