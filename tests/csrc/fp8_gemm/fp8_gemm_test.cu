#include <assert.h>
#include <cublas_v2.h>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "src/turbomind/layers/DenseWeight.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/gemm.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

// CUTLASS FP8 Gemm
#include "src/turbomind/kernels/cutlass_w8a8/scaled_mm_entry.h"

// cuBLAS FP8 Gemm
#include "src/turbomind/utils/cublasFP8MMWrapper.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"

#include "test_utils.h"

using namespace turbomind;
namespace ft = turbomind;

// construct mnk
std::vector<int> NUM_TOKENS   = {1, 2, 4};
std::vector<int> INTER_SIZE   = {5120};
std::vector<int> HIDDEN_SIZES = {5120};

// Can be replaced by the function provided by a test framework
class TestFailureError: public std::exception {
private:
    std::string msg_;

public:
    explicit TestFailureError() = default;
    explicit TestFailureError(std::string name, std::string msg = "")
    {
        msg_ = fmtstr("TEST FAIL [%s] %s", name.c_str(), msg.c_str());
    }
    const char* what() const throw()
    {
        return msg_.c_str();
    }
};

#define EXPECT_TRUE(cond)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            TM_LOG_ERROR("TEST FAIL [%s] at %s:%d", __func__, __FILE__, __LINE__);                                     \
            throw TestFailureError(__func__);                                                                          \
        }                                                                                                              \
    } while (false)

#define EXPECT_ALMOST_EQUAL(name, dtype, ctype, out, ref, atol, rtol)                                                  \
    do {                                                                                                               \
        bool is_ok = checkResult<dtype, ctype>(name, out, ref, atol, rtol);                                            \
        if (!is_ok) {                                                                                                  \
            TM_LOG_ERROR("TEST FAIL [%s] at %s:%d", __func__, __FILE__, __LINE__);                                     \
            throw TestFailureError(__func__);                                                                          \
        }                                                                                                              \
    } while (false)

template<DataType computeType>
void computeReference(GemmOp         transa,
                      GemmOp         transb,
                      TensorWrapper& C,
                      TensorWrapper& A,
                      TensorWrapper& B,
                      float          alpha = 1.0f,
                      float          beta  = 0.0f)
{
    size_t m = C.shape[0];
    size_t n = C.shape[1];
    size_t k = A.shape[1];

    size_t lda = (transa == GEMM_OP_N) ? k : m;
    size_t ldb = (transb == GEMM_OP_N) ? n : k;
    size_t ldc = n;

    cudaDataType_t atype        = (A.type == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t btype        = (B.type == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t ctype        = (C.type == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t compute_type = (computeType == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));

    half        h_alpha = (half)alpha;
    half        h_beta  = (half)beta;
    const void* _alpha  = (computeType == TYPE_FP16) ? (const void*)&h_alpha : (const void*)&alpha;
    const void* _beta   = (computeType == TYPE_FP16) ? (const void*)&h_beta : (const void*)&beta;

    check_cuda_error(cublasGemmEx(cublas_handle,
                                  getCublasOperation(transb),
                                  getCublasOperation(transa),
                                  n,
                                  m,
                                  k,
                                  _alpha,
                                  (const void*)B.data,
                                  btype,
                                  ldb,
                                  (const void*)A.data,
                                  atype,
                                  lda,
                                  _beta,
                                  (void*)C.data,
                                  ctype,
                                  ldc,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT));
    check_cuda_error(cublasDestroy(cublas_handle));
    cudaDeviceSynchronize();
}

template<typename T>
std::string toString()
{
    std::string str = "dtype=";
    str += std::is_same<T, float>::value ? "FP32" : "FP16";
    return str;
}

template<typename T, DataType ctype>
std::string toString()
{
    std::string str = "dtype=";
    str += std::is_same<T, float>::value ? "FP32" : "FP16";
    str += ", compute_type=";
    str += (ctype == TYPE_FP32) ? "FP32" : "FP16";
    return str;
}

std::string toString(GemmOp op)
{
    return op == GEMM_OP_N ? "N" : "T";
}

struct GemmOpPair {
    GemmOp transa;
    GemmOp transb;
};

/// Align to cublasFP8MMWrapper， Only support NT
static const std::vector<GemmOpPair> op_pairs{
    /*{GEMM_OP_N, GEMM_OP_N},*/ {GEMM_OP_N, GEMM_OP_T} /*, {GEMM_OP_T, GEMM_OP_N}, {GEMM_OP_T, GEMM_OP_T}*/};

static inline std::string getTestName(const char* func_name, GemmOp transa, GemmOp transb, size_t m, size_t n, size_t k)
{
    return fmtstr("%s [opA=%s, opB=%s, m=%ld, n=%ld, k=%ld]",
                  func_name,
                  getGemmOpString(transa).c_str(),
                  getGemmOpString(transb).c_str(),
                  m,
                  n,
                  k);
}

static inline std::string getTestName(const char* func_name, GemmOpPair op_pairs, size_t m, size_t n, size_t k)
{
    return getTestName(func_name, op_pairs.transa, op_pairs.transb, m, n, k);
}

/////////////////////////////////// Unittests //////////////////////////////////////////

template<typename T, DataType computeType>
void testFP8GemmCorrectnessCUTLASS(size_t m, size_t n, size_t k, bool profiling)
{
    // Test Correctness between Matmul and CUTLASS FP8

    TM_LOG_INFO(
        "CUTLASS function correctness test [m=%ld, n=%ld, k=%ld, %s]", m, n, k, toString<T, computeType>().c_str());

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA> allocator(getDevice());

    DataType dtype = getTensorType<T>();

    // Construct FP16 Tensor
    TensorWrapper input_tensor(&allocator, dtype, {m, k}, false);
    TensorWrapper weight_tensor(&allocator, dtype, {k, n}, false);
    TensorWrapper result_fp16(&allocator, dtype, {m, n}, true);

    // Construct FP8 Tensor
    TensorWrapper q_input_tensor(&allocator, TYPE_FP8_E4M3, {m, k}, false);
    TensorWrapper q_weight_tensor(&allocator, TYPE_FP8_E4M3, {k, n}, false);
    TensorWrapper result_fp8(&allocator, dtype, {m, n}, true);
    float         input_scale  = 0.007547169923782349;
    float         weight_scale = 0.012519561685621738;

    float input_scale_inv  = 1.0 / input_scale;
    float weight_scale_inv = 1.0 / weight_scale;

    float *g_input_scale, *g_input_scale_inv   = nullptr;
    float *g_weight_scale, *g_weight_scale_inv = nullptr;

    cudaMalloc((void**)&g_input_scale, sizeof(float));
    cudaMalloc((void**)&g_weight_scale, sizeof(float));
    cudaMemcpy(g_input_scale, &input_scale, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight_scale, &weight_scale, sizeof(float) * 1, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&g_input_scale_inv, sizeof(float));
    cudaMalloc((void**)&g_weight_scale_inv, sizeof(float));
    invokeConvertWeightToInv(g_input_scale_inv, g_input_scale, 1, stream);
    invokeConvertWeightToInv(g_weight_scale_inv, g_weight_scale, 1, stream);
    
    invokeScaleFP8QuantMatrix<T, QUANTIZE_MODE::PER_TENSOR>(
        reinterpret_cast<__nv_fp8_e4m3*>(q_input_tensor.data),
        g_input_scale_inv,
        reinterpret_cast<T*>(input_tensor.data),
        m,
        m*k,
        stream);
    
    invokeScaleFP8QuantMatrix<T, QUANTIZE_MODE::PER_TENSOR>(
        reinterpret_cast<__nv_fp8_e4m3*>(q_weight_tensor.data),
        g_weight_scale_inv,
        reinterpret_cast<T*>(weight_tensor.data),
        k,
        k*n,
        stream);
    
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
    std::mutex*   cublas_wrapper_mutex = new std::mutex();

    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_fp16(new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, &allocator));
    cudaDataType_t                       cuda_dtype = std::is_same<float, T>::value ? CUDA_R_32F : CUDA_R_16F;  // dtype
    cudaDataType_t cuda_ctype = (DataType::TYPE_FP32 == computeType) ? CUDA_R_32F : CUDA_R_16F;  // compute type
    cublas_wrapper_fp16->setGemmConfig(cuda_dtype, cuda_dtype, cuda_dtype, cuda_ctype);

    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_fp8(new ft::cublasFP8MMWrapper(
        cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, &allocator));
    cublas_wrapper_fp8->setFP8GemmConfig();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const float atol = 20;
    const float rtol = 1e-1f;

    for (auto& op_pair : op_pairs) {
        std::string tc_name = getTestName(__func__, op_pair, m, n, k);

        // Switch A/B because Gemm expects column major layout as cublas does.
        size_t lda = (op_pair.transa == GEMM_OP_N) ? k : m;
        size_t ldb = (op_pair.transb == GEMM_OP_N) ? n : k;
        size_t ldc = n;

        cublas_wrapper_fp16->Gemm(getCublasOperation(op_pair.transb),
                                  getCublasOperation(op_pair.transa),
                                  n,
                                  m,
                                  k,
                                  weight_tensor.data,
                                  ldb,
                                  input_tensor.data,
                                  lda,
                                  result_fp16.data,
                                  ldc);

        cutlass_scaled_mm(reinterpret_cast<half*>(result_fp8.data),
                          (int)1,  // batch_count
                          (int)m,
                          (int)n,
                          (int)k,
                          (int64_t)0,
                          (int64_t)0,
                          (int64_t)0,
                          &alpha,
                          &beta,
                          reinterpret_cast<__nv_fp8_e4m3*>(q_input_tensor.data),
                          reinterpret_cast<__nv_fp8_e4m3*>(q_weight_tensor.data),
                          g_input_scale,
                          g_weight_scale,
                          stream);

        if (profiling) {
            std::vector<float> times;
            int                iter = 1000;

            while (iter--) {
                TIME_MS_START_TOOL(cutlass, stream)
                cutlass_scaled_mm(reinterpret_cast<half*>(result_fp8.data),
                                  (int)1,  // batch_count
                                  (int)m,
                                  (int)n,
                                  (int)k,
                                  (int64_t)0,
                                  (int64_t)0,
                                  (int64_t)0,
                                  &alpha,
                                  &beta,
                                  reinterpret_cast<__nv_fp8_e4m3*>(q_input_tensor.data),
                                  reinterpret_cast<__nv_fp8_e4m3*>(q_weight_tensor.data),
                                  g_input_scale,
                                  g_weight_scale,
                                  stream);
                TIME_MS_END_TOOL(cutlass, stream)
                times.push_back(ms_cutlass * 1000);
            }

            float mean = 0.0, stdev = 0.0;
            analysisTimes(times, mean, stdev);
            printf("cutlass_scaled_mm, mean: %f us, stdev: %f\n", mean, stdev);
        }

        // EXPECT_ALMOST_EQUAL(tc_name + " cublas_fp16_vs_fp8", T, computeType, result_fp16, result_fp8, atol, rtol);

        compareTwoTensor(reinterpret_cast<half*>(result_fp8.data), reinterpret_cast<half*>(result_fp16.data), m * n);

        float cs = CosineSimilarity(
            reinterpret_cast<half*>(result_fp8.data), reinterpret_cast<half*>(result_fp16.data), m * n);
        printf("CosineSimilarity: %f\n", cs);
    }

    delete cublas_wrapper_mutex;
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cudaStreamDestroy(stream));

    cudaFree(g_input_scale);
    cudaFree(g_weight_scale);
    cudaFree(g_input_scale_inv);
    cudaFree(g_weight_scale_inv);
}

template<typename T, DataType computeType>
void testFP8GemmAccuracy(size_t m, size_t n, size_t k, bool profiling)
{
    TM_LOG_INFO(
        "Matmul function correctness test [m=%ld, n=%ld, k=%ld, %s]", m, n, k, toString<T, computeType>().c_str());

    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    Allocator<AllocatorType::CUDA> allocator(getDevice());

    DataType dtype = getTensorType<T>();

    // Construct FP16 Tensor
    TensorWrapper input_tensor(&allocator, dtype, {m, k}, false);
    TensorWrapper weight_tensor(&allocator, dtype, {k, n}, false);
    TensorWrapper result_fp16(&allocator, dtype, {m, n}, true);

    // Construct FP8 Tensor
    TensorWrapper q_input_tensor(&allocator, TYPE_FP8_E4M3, {m, k}, false);
    TensorWrapper q_weight_tensor(&allocator, TYPE_FP8_E4M3, {k, n}, false);
    TensorWrapper result_fp8(&allocator, dtype, {m, n}, true);
    float         input_scale  = 0.007547169923782349;
    float         weight_scale = 0.012519561685621738;

    float input_scale_inv  = 1.0 / input_scale;
    float weight_scale_inv = 1.0 / weight_scale;

    float *g_input_scale, *g_input_scale_inv   = nullptr;
    float *g_weight_scale, *g_weight_scale_inv = nullptr;

    cudaMalloc((void**)&g_input_scale, sizeof(float));
    cudaMalloc((void**)&g_weight_scale, sizeof(float));
    cudaMemcpy(g_input_scale, &input_scale, sizeof(float) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight_scale, &weight_scale, sizeof(float) * 1, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&g_input_scale_inv, sizeof(float));
    cudaMalloc((void**)&g_weight_scale_inv, sizeof(float));
    invokeConvertWeightToInv(g_input_scale_inv, g_input_scale, 1, stream);
    invokeConvertWeightToInv(g_weight_scale_inv, g_weight_scale, 1, stream);
    
    invokeScaleFP8QuantMatrix<T, QUANTIZE_MODE::PER_TENSOR>(
        reinterpret_cast<__nv_fp8_e4m3*>(q_input_tensor.data),
        g_input_scale_inv,
        reinterpret_cast<T*>(input_tensor.data),
        m,
        m*k,
        stream);
    
    invokeScaleFP8QuantMatrix<T, QUANTIZE_MODE::PER_TENSOR>(
        reinterpret_cast<__nv_fp8_e4m3*>(q_weight_tensor.data),
        g_weight_scale_inv,
        reinterpret_cast<T*>(weight_tensor.data),
        k,
        k*n,
        stream);

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
    std::mutex*   cublas_wrapper_mutex = new std::mutex();

    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_fp16(new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, &allocator));
    cudaDataType_t                       cuda_dtype = std::is_same<float, T>::value ? CUDA_R_32F : CUDA_R_16F;  // dtype
    cudaDataType_t cuda_ctype = (DataType::TYPE_FP32 == computeType) ? CUDA_R_32F : CUDA_R_16F;  // compute type
    cublas_wrapper_fp16->setGemmConfig(cuda_dtype, cuda_dtype, cuda_dtype, cuda_ctype);

    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_fp8(new ft::cublasFP8MMWrapper(
        cublas_handle, cublaslt_handle, stream, &cublas_algo_map, cublas_wrapper_mutex, &allocator));
    cublas_wrapper_fp8->setFP8GemmConfig();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const float atol = 20;
    const float rtol = 1e-1f;
    for (auto& op_pair : op_pairs) {
        std::string tc_name = getTestName(__func__, op_pair, m, n, k);

        // Switch A/B because Gemm expects column major layout as cublas does.
        size_t lda = (op_pair.transa == GEMM_OP_N) ? k : m;
        size_t ldb = (op_pair.transb == GEMM_OP_N) ? n : k;
        size_t ldc = n;

        cublas_wrapper_fp16->Gemm(getCublasOperation(op_pair.transb),
                                  getCublasOperation(op_pair.transa),
                                  n,
                                  m,
                                  k,
                                  weight_tensor.data,
                                  ldb,
                                  input_tensor.data,
                                  lda,
                                  result_fp16.data,
                                  ldc);

        reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_fp8.get())
            ->Gemm(reinterpret_cast<half*>(result_fp8.data),
                   (int)1,
                   (int)m,
                   (int)n,
                   (int)k,
                   (int64_t)0,
                   (int64_t)0,
                   (int64_t)0,
                   &alpha,
                   &beta,
                   reinterpret_cast<const __nv_fp8_e4m3*>(q_input_tensor.data),
                   reinterpret_cast<__nv_fp8_e4m3*>(q_weight_tensor.data),
                   g_input_scale,
                   g_weight_scale,
                   stream);

        if (profiling) {
            std::vector<float> times;
            int                iter = 1000;

            while (iter--) {
                TIME_MS_START_TOOL(cublass_fp16, stream)

                cublas_wrapper_fp16->Gemm(getCublasOperation(op_pair.transb),
                                          getCublasOperation(op_pair.transa),
                                          n,
                                          m,
                                          k,
                                          weight_tensor.data,
                                          ldb,
                                          input_tensor.data,
                                          lda,
                                          result_fp16.data,
                                          ldc);

                TIME_MS_END_TOOL(cublass_fp16, stream)
                times.push_back(ms_cublass_fp16 * 1000);
            }

            float mean = 0.0, stdev = 0.0;
            analysisTimes(times, mean, stdev);
            printf("cutlass_fp16, mean: %f us, stdev: %f\n", mean, stdev);
        }

        if (profiling) {
            std::vector<float> times;
            int                iter = 1000;

            while (iter--) {
                TIME_MS_START_TOOL(cublass, stream)

                reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_fp8.get())
                    ->Gemm(reinterpret_cast<half*>(result_fp8.data),
                           (int)1,
                           (int)m,
                           (int)n,
                           (int)k,
                           (int64_t)0,
                           (int64_t)0,
                           (int64_t)0,
                           &alpha,
                           &beta,
                           reinterpret_cast<const __nv_fp8_e4m3*>(q_input_tensor.data),
                           reinterpret_cast<__nv_fp8_e4m3*>(q_weight_tensor.data),
                           g_input_scale,
                           g_weight_scale,
                           stream);
                TIME_MS_END_TOOL(cublass, stream)
                times.push_back(ms_cublass * 1000);
            }

            float mean = 0.0, stdev = 0.0;
            analysisTimes(times, mean, stdev);
            printf("cutlass_scaled_mm, mean: %f us, stdev: %f\n", mean, stdev);
        }

        // EXPECT_ALMOST_EQUAL(tc_name + " cublas_fp16_vs_fp8", T, computeType, result_fp16, result_fp8, atol, rtol);

        compareTwoTensor(reinterpret_cast<half*>(result_fp8.data), reinterpret_cast<half*>(result_fp16.data), m * n);

        float cs = CosineSimilarity(
            reinterpret_cast<half*>(result_fp8.data), reinterpret_cast<half*>(result_fp16.data), m * n);
        printf("CosineSimilarity: %f\n", cs);
    }

    delete cublas_wrapper_mutex;
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cudaStreamDestroy(stream));

    cudaFree(g_input_scale);
    cudaFree(g_weight_scale);
    cudaFree(g_input_scale_inv);
    cudaFree(g_weight_scale_inv);
}

int main(int argc, char* argv[])
{
    // testGemmCreate();
    using testcase_t = std::tuple<size_t, size_t, size_t>;
    std::vector<testcase_t> testcases;
    for (auto m : NUM_TOKENS)
        for (auto n : INTER_SIZE)
            for (auto k : HIDDEN_SIZES)
                testcases.push_back({m, n, k});

    TM_LOG_INFO("==== Begin Test FP8 Gemm ====");

    // Computation correctness tests
    for (testcase_t& tc : testcases) {
        size_t m = std::get<0>(tc);
        size_t n = std::get<1>(tc);
        size_t k = std::get<2>(tc);

        bool profiling = true;
        testFP8GemmCorrectnessCUTLASS<half, TYPE_FP32>(m, n, k, profiling);
        testFP8GemmAccuracy<half, TYPE_FP32>(m, n, k, profiling);
        printf("\n");
    }

    TM_LOG_INFO("==== Test done ====");
    return 0;
}
