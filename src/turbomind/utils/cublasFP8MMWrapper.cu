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

#include "cublasFP8MMWrapper.h"
#include "cuda_utils.h"

#ifdef ENABLE_FP8
namespace turbomind {

#define CUBLAS_WORKSPACE_1MB 1048576
cublasFP8MMWrapper::cublasFP8MMWrapper(cublasLtHandle_t cublaslt_handle,
                                       cudaStream_t     stream,
                                       cublasAlgoMap*   cublas_algo_map,
                                       std::mutex*      mu,
                                       void*            cublas_workspace,
                                       void*            cublas_workspace_qgemm):
    cublasMMWrapper(nullptr, cublaslt_handle, stream, cublas_algo_map, mu, cublas_workspace)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK_WITH_INFO(cublas_workspace != nullptr, "must pass cublas_workspace to cublasFP8MMWrapper");
    cublasVersionCheck();

    cublas_workspace_qgemm_ = cublas_workspace_qgemm;
}

cublasFP8MMWrapper::cublasFP8MMWrapper(cublasHandle_t   cublas_handle,
                                       cublasLtHandle_t cublaslt_handle,
                                       cudaStream_t     stream,
                                       cublasAlgoMap*   cublas_algo_map,
                                       std::mutex*      mu,
                                       void*            cublas_workspace,
                                       void*            cublas_workspace_qgemm):
    cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, mu, cublas_workspace)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK_WITH_INFO(cublas_workspace != nullptr, "must pass cublas_workspace to cublasFP8MMWrapper");
    cublasVersionCheck();

    cublas_workspace_qgemm_ = cublas_workspace_qgemm;
}

cublasFP8MMWrapper::~cublasFP8MMWrapper()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_ = nullptr;
}

void cublasFP8MMWrapper::cublasVersionCheck()
{
    cublasGetProperty(MAJOR_VERSION, &version_major_);
    cublasGetProperty(MINOR_VERSION, &version_minor_);
    cublasGetProperty(PATCH_LEVEL, &version_patch_);
    size_t cublasVersion = (version_major_ * 10000 + version_minor_ * 100 + version_patch_);
#if defined(FP8_MHA) || !defined(FP8_GEMM_OUTPUT_QUANT_DISABLE)
    FT_CHECK_WITH_INFO((version_major_ > 11) || (version_major_ == 11 && version_minor_ == 11 && version_patch_ >= 4),
                       "FP8 MHA needs d-scale, which is only supported after cublas 11.11.4 !");

#endif
}

void cublasFP8MMWrapper::Gemm(__nv_bfloat16*       res,
                              int                  batchCount,
                              int                  m,
                              int                  n,
                              int                  k,
                              int64_t              strideA,
                              int64_t              strideB,
                              int64_t              strideD,
                              const float*         alpha,
                              const float*         beta,
                              const __nv_fp8_e4m3* input,
                              const __nv_fp8_e4m3* kernel,
                              const float*         input_scale,
                              const float*         kernel_scale)
{
    Gemm(res,
         batchCount,
         m,
         n,
         k,
         strideA,
         strideB,
         strideD,
         alpha,
         beta,
         input,
         kernel,
         input_scale,
         kernel_scale,
         (cudaStream_t)0);
}

void cublasFP8MMWrapper::Gemm(__nv_bfloat16*       res,
                              int                  batchCount,
                              int                  m,
                              int                  n,
                              int                  k,
                              int64_t              strideA,
                              int64_t              strideB,
                              int64_t              strideD,
                              const float*         alpha,
                              const float*         beta,
                              const __nv_fp8_e4m3* input,
                              const __nv_fp8_e4m3* kernel,
                              const float*         input_scale,
                              const float*         kernel_scale,
                              cudaStream_t         stream,
                              bool                 fastAccum)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_->lock();

    const void*  devAscalePtr = (const void*)kernel_scale;
    const void*  devBscalePtr = (const void*)input_scale;
    const size_t wsSizeBytes  = CUBLAS_WORKSPACE_SIZE;

    const auto aType       = CUDA_R_8F_E4M3;
    const auto bType       = CUDA_R_8F_E4M3;
    const auto dType       = CUDA_R_16BF;
    const auto computeType = CUBLAS_COMPUTE_32F;
    const auto scaleType   = CUDA_R_32F;
    // const auto epilogueAuxType = CUDA_R_16BF;

    const cublasOperation_t tA = CUBLAS_OP_T;
    const cublasOperation_t tB = CUBLAS_OP_N;

    //------- init, desc & tensors
    cublasLtMatmulDesc_t   matmulDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Ddesc;

    {
        check_cuda_error(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
        check_cuda_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));

        if (version_major_ >= 11 && version_minor_ >= 11 && version_patch_ > 0 && fastAccum) {
            const int8_t fastAccuMode = 1;  // enable fast imprecise accum
            check_cuda_error(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(decltype(fastAccuMode))));
        }

        // TODO: Check that do we need to set these attributes
        // TODO: comment them for compiler first
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &devAscalePtr, sizeof(devAscalePtr)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &devBscalePtr, sizeof(devBscalePtr)));
    }

    {
        const int64_t lda = k;
        const int64_t ldb = k;
        const int64_t ldd = n;

        // create matrix descriptors, we are good with the details here so no need
        // to set any extra attributes
        check_cuda_error(
            cublasLtMatrixLayoutCreate(&Adesc, aType, tA == CUBLAS_OP_N ? n : k, tA == CUBLAS_OP_N ? k : n, lda));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        }

        check_cuda_error(
            cublasLtMatrixLayoutCreate(&Bdesc, bType, tB == CUBLAS_OP_N ? k : m, tB == CUBLAS_OP_N ? m : k, ldb));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        }

        check_cuda_error(cublasLtMatrixLayoutCreate(&Ddesc, dType, n, m, ldd));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
        }
    }

    bool                    findAlgo = cublas_algo_map_->isExist(batchCount, n, m, k, FP8_DATATYPE);
    cublasLtMatmulAlgo_info info     = cublas_algo_map_->getAlgo(batchCount, n, m, k, FP8_DATATYPE);
    if (info.stages == -1) {
        findAlgo = false;
    }

    cublasLtMatmulAlgo_t algo;
    int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo) {
        if (info.workspaceSize > workspaceSize) {
            findAlgo = false;
        }
        else {
            cublasLtMatmulAlgoInit(
                cublaslt_handle_, computeType, scaleType, aType, bType, dType, dType, info.algoId, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme));

#if (CUDART_VERSION >= 11000)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));
#endif

#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId));
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &(info.mma_shapeId), sizeof(info.mma_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &(info.cga_shapeId), sizeof(info.cga_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &(info.sche_mode), sizeof(info.sche_mode));
#endif
        }
    }

    {
        cublasStatus_t status = cublasLtMatmul(cublaslt_handle_,
                                               matmulDesc,
                                               alpha,
                                               kernel,
                                               Adesc,
                                               input,
                                               Bdesc,
                                               beta,
                                               nullptr,  // Cptr, not used here
                                               Ddesc,
                                               res,
                                               Ddesc,
                                               (findAlgo ? (&algo) : NULL),
                                               cublas_workspace_,
                                               wsSizeBytes,
                                               stream);
        check_cuda_error(status);
    }

    if (Ddesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Ddesc));
    }
    if (Bdesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Bdesc));
    }
    if (Adesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Adesc));
    }
    if (matmulDesc) {
        check_cuda_error(cublasLtMatmulDescDestroy(matmulDesc));
    }

    mu_->unlock();
}

void cublasFP8MMWrapper::Gemm(__nv_fp8_e4m3*       res,
                              int                  batchCount,
                              int                  m,
                              int                  n,
                              int                  k,
                              int64_t              strideA,
                              int64_t              strideB,
                              int64_t              strideD,
                              const float*         alpha,
                              const float*         beta,
                              const __nv_fp8_e4m3* input,
                              const __nv_fp8_e4m3* kernel,
                              const float*         input_scale,
                              const float*         kernel_scale,
                              const float*         output_scale)
{
    Gemm(res,
         batchCount,
         m,
         n,
         k,
         strideA,
         strideB,
         strideD,
         alpha,
         beta,
         input,
         kernel,
         input_scale,
         kernel_scale,
         output_scale,
         0);
}

void cublasFP8MMWrapper::Gemm(__nv_fp8_e4m3*       res,
                              int                  batchCount,
                              int                  m,
                              int                  n,
                              int                  k,
                              int64_t              strideA,
                              int64_t              strideB,
                              int64_t              strideD,
                              const float*         alpha,
                              const float*         beta,
                              const __nv_fp8_e4m3* input,
                              const __nv_fp8_e4m3* kernel,
                              const float*         input_scale,
                              const float*         kernel_scale,
                              const float*         output_scale,
                              cudaStream_t         stream,
                              bool                 fastAccum)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_->lock();

    const void* devAscalePtr = (const void*)kernel_scale;
    const void* devBscalePtr = (const void*)input_scale;
    const void* devDscalePtr = (const void*)output_scale;

    FT_CHECK(cublas_workspace_ != nullptr);
    const size_t wsSizeBytes = CUBLAS_WORKSPACE_SIZE;

    const auto aType       = CUDA_R_8F_E4M3;
    const auto bType       = CUDA_R_8F_E4M3;
    const auto cType       = CUDA_R_16BF;
    const auto dType       = CUDA_R_8F_E4M3;
    const auto computeType = CUBLAS_COMPUTE_32F;
    const auto scaleType   = CUDA_R_32F;

    const cublasOperation_t tA = CUBLAS_OP_T;
    const cublasOperation_t tB = CUBLAS_OP_N;

    //------- init, desc & tensors
    cublasLtMatmulDesc_t   matmulDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cublasLtMatrixLayout_t Ddesc;

    {
        check_cuda_error(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
        check_cuda_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));

        if (version_major_ >= 11 && version_minor_ >= 11 && version_patch_ > 0 && fastAccum) {
            const int8_t fastAccuMode = 1;  // enable fast imprecise accum
            check_cuda_error(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(decltype(fastAccuMode))));
        }

        // TODO: Check that do we need to set these attributes
        // TODO: comment them for compiler first
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &devAscalePtr, sizeof(devAscalePtr)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &devBscalePtr, sizeof(devBscalePtr)));
        // check_cuda_error(cublasLtMatmulDescSetAttribute(
        //     matmulDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &devDscalePtr, sizeof(devDscalePtr)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &devDscalePtr, sizeof(devDscalePtr)));
    }

    {
        const int64_t lda = k;
        const int64_t ldb = k;
        const int64_t ldd = n;

        // create matrix descriptors, we are good with the details here so no need
        // to set any extra attributes
        check_cuda_error(
            cublasLtMatrixLayoutCreate(&Adesc, aType, tA == CUBLAS_OP_N ? n : k, tA == CUBLAS_OP_N ? k : n, lda));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        }

        check_cuda_error(
            cublasLtMatrixLayoutCreate(&Bdesc, bType, tB == CUBLAS_OP_N ? k : m, tB == CUBLAS_OP_N ? m : k, ldb));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        }

        check_cuda_error(cublasLtMatrixLayoutCreate(&Cdesc, cType, n, m, ldd));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
        }
        check_cuda_error(cublasLtMatrixLayoutCreate(&Ddesc, dType, n, m, ldd));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
        }
    }

    bool                    findAlgo = cublas_algo_map_->isExist(batchCount, n, m, k, FP8_DATATYPE);
    cublasLtMatmulAlgo_info info     = cublas_algo_map_->getAlgo(batchCount, n, m, k, FP8_DATATYPE);
    if (info.stages == -1) {
        findAlgo = false;
    }

    cublasLtMatmulAlgo_t algo;
    int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo) {
        if (info.workspaceSize > workspaceSize) {
            findAlgo = false;
        }
        else {
            cublasLtMatmulAlgoInit(
                cublaslt_handle_, computeType, scaleType, aType, bType, cType, dType, info.algoId, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme));

#if (CUDART_VERSION >= 11000)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));
#endif

#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId));
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &(info.mma_shapeId), sizeof(info.mma_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &(info.cga_shapeId), sizeof(info.cga_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &(info.sche_mode), sizeof(info.sche_mode));
#endif
        }
    }

    {
        cublasStatus_t status = cublasLtMatmul(cublaslt_handle_,
                                               matmulDesc,
                                               alpha,
                                               kernel,
                                               Adesc,
                                               input,
                                               Bdesc,
                                               beta,
                                               nullptr,  // Cptr, not used here
                                               Cdesc,
                                               res,
                                               Ddesc,
                                               (findAlgo ? (&algo) : NULL),
                                               cublas_workspace_,
                                               wsSizeBytes,
                                               stream);
        check_cuda_error(status);
    }

    if (Ddesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Ddesc));
    }
    if (Cdesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Cdesc));
    }
    if (Bdesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Bdesc));
    }
    if (Adesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Adesc));
    }
    if (matmulDesc) {
        check_cuda_error(cublasLtMatmulDescDestroy(matmulDesc));
    }

    mu_->unlock();
}


void cublasFP8MMWrapper::Gemm(half*                res,
                              int                  batchCount,
                              int                  m,
                              int                  n,
                              int                  k,
                              int64_t              strideA,
                              int64_t              strideB,
                              int64_t              strideD,
                              const float*         alpha,
                              const float*         beta,
                              const __nv_fp8_e4m3* input,
                              const __nv_fp8_e4m3* kernel,
                              const float*         input_scale,
                              const float*         kernel_scale)
{
    Gemm(res,
         batchCount,
         m,
         n,
         k,
         strideA,
         strideB,
         strideD,
         alpha,
         beta,
         input,
         kernel,
         input_scale,
         kernel_scale,
         (cudaStream_t)0);
}

void cublasFP8MMWrapper::Gemm(half*                res,
                              int                  batchCount,
                              int                  m,  // b
                              int                  n,  // o
                              int                  k,  // i
                              int64_t              strideA,
                              int64_t              strideB,
                              int64_t              strideD,
                              const float*         alpha,
                              const float*         beta,
                              const __nv_fp8_e4m3* input,
                              const __nv_fp8_e4m3* kernel,
                              const float*         input_scale,
                              const float*         kernel_scale,
                              cudaStream_t         stream,
                              bool                 fastAccum)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mu_->lock();

    const void*  devAscalePtr = (const void*)kernel_scale;
    const void*  devBscalePtr = (const void*)input_scale;
    const size_t wsSizeBytes  = CUBLAS_WORKSPACE_SIZE;

    const auto aType       = CUDA_R_8F_E4M3;
    const auto bType       = CUDA_R_8F_E4M3;
    const auto dType       = CUDA_R_16F;
    const auto computeType = CUBLAS_COMPUTE_32F;
    const auto scaleType   = CUDA_R_32F;

    const cublasOperation_t tA = CUBLAS_OP_T;
    const cublasOperation_t tB = CUBLAS_OP_N;

    //------- init, desc & tensors
    cublasLtMatmulDesc_t   matmulDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Ddesc;

    {
        check_cuda_error(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
        check_cuda_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));

        if (version_major_ >= 11 && version_minor_ >= 11 && version_patch_ > 0 && fastAccum) {
            const int8_t fastAccuMode = 1;  // enable fast imprecise accum
            check_cuda_error(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(decltype(fastAccuMode))));
        }

        // TODO: Check that do we need to set these attributes
        // TODO: comment them for compiler first
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &devAscalePtr, sizeof(devAscalePtr)));
        check_cuda_error(cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &devBscalePtr, sizeof(devBscalePtr)));
    }

    {
        const int64_t lda = k; // i
        const int64_t ldb = k; // i
        const int64_t ldd = n; // o

        // create matrix descriptors, we are good with the details here so no need
        // to set any extra attributes
        check_cuda_error(
            cublasLtMatrixLayoutCreate(&Adesc, aType, tA == CUBLAS_OP_N ? n : k, tA == CUBLAS_OP_N ? k : n, lda));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        }

        check_cuda_error(
            cublasLtMatrixLayoutCreate(&Bdesc, bType, tB == CUBLAS_OP_N ? k : m, tB == CUBLAS_OP_N ? m : k, ldb));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        }

        check_cuda_error(cublasLtMatrixLayoutCreate(&Ddesc, dType, n, m, ldd));
        if (batchCount > 1) {
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
            check_cuda_error(cublasLtMatrixLayoutSetAttribute(
                Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
        }
    }

    bool                    findAlgo = cublas_algo_map_->isExist(batchCount, n, m, k, FP8_DATATYPE);
    cublasLtMatmulAlgo_info info     = cublas_algo_map_->getAlgo(batchCount, n, m, k, FP8_DATATYPE);
    if (info.stages == -1) {
        findAlgo = false;
    }

    cublasLtMatmulAlgo_t algo;
    int                  workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo) {
        if (info.workspaceSize > workspaceSize) {
            findAlgo = false;
        }
        else {
            cublasLtMatmulAlgoInit(
                cublaslt_handle_, computeType, scaleType, aType, bType, dType, dType, info.algoId, &algo);
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(info.reductionScheme));

#if (CUDART_VERSION >= 11000)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));
#endif

#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId), sizeof(info.inner_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &(info.cluster_shapeId), sizeof(info.cluster_shapeId));
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &(info.mma_shapeId), sizeof(info.mma_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &(info.cga_shapeId), sizeof(info.cga_shapeId));
            cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &(info.sche_mode), sizeof(info.sche_mode));
#endif
        }
    }

    {
        cublasStatus_t status = cublasLtMatmul(cublaslt_handle_,
                                               matmulDesc,
                                               alpha,
                                               kernel,
                                               Adesc,
                                               input,
                                               Bdesc,
                                               beta,
                                               nullptr,  // Cptr, not used here
                                               Ddesc,
                                               res,
                                               Ddesc,
                                               (findAlgo ? (&algo) : NULL),
                                               cublas_workspace_,
                                               wsSizeBytes,
                                               stream);
        check_cuda_error(status);
    }

    if (Ddesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Ddesc));
    }
    if (Bdesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Bdesc));
    }
    if (Adesc) {
        check_cuda_error(cublasLtMatrixLayoutDestroy(Adesc));
    }
    if (matmulDesc) {
        check_cuda_error(cublasLtMatmulDescDestroy(matmulDesc));
    }

    mu_->unlock();
}

}  // namespace turbomind

#endif // #ifdef ENABLE_FP8