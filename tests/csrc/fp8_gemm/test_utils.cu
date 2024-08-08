
#include "test_utils.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <numeric>

/// Compute performance in GFLOP/s
double gflops(const int m, const int n, const int k, const double runtime_s)
{
  // Two flops per multiply-add
  uint64_t flop = uint64_t(2) * m * n * k;
  double gflop = double(flop) / double(1.0e9);
  return gflop / runtime_s;
}

bool almostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b)) {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

template<typename T>
bool _checkResult(std::string name, TensorWrapper& out, TensorWrapper& ref, float atol, float rtol)
{
    assert(out.type == ref.type);

    size_t out_size = out.size();
    size_t ref_size = ref.size();
    T*     h_out    = reinterpret_cast<T*>(malloc(sizeof(T) * out_size));
    T*     h_ref    = reinterpret_cast<T*>(malloc(sizeof(T) * ref_size));

    cudaMemcpy(h_out, out.data, sizeof(T) * out_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, ref.data, sizeof(T) * ref_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    size_t failures = 0;
    for (size_t i = 0; i < out_size; ++i) {
        // The values for the output and the reference.
        float a = (float)h_out[i];
        float b = (float)h_ref[i];

        bool ok = almostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4) {
            TM_LOG_ERROR(">> invalid result for i=%lu:", i);
            TM_LOG_ERROR(">>    found......: %10.6f", a);
            TM_LOG_ERROR(">>    expected...: %10.6f", b);
            TM_LOG_ERROR(">>    error......: %.6f", fabsf(a - b));
            TM_LOG_ERROR(">>    tol........: %.6f", atol + rtol * fabs(b));
        }

        // Update the number of failures.
        failures += ok ? 0 : 1;
    }

    // Allow not matched up to 1% elements.
    size_t tol_failures = (size_t)(0.01 * out_size);
    TM_LOG_INFO("check....... %30s : %s (failures: %.2f%% atol: %.2e rtol: %.2e)",
                name.c_str(),
                failures <= tol_failures ? "OK" : "FAILED",
                100. * failures / out_size,
                atol,
                rtol);
    return failures <= tol_failures;
}

template<typename T, DataType computeType>
bool checkResult(std::string name, TensorWrapper& out, TensorWrapper& ref, float atol=1e-6f, float rtol=1e-4f)
{
    //float atol  = (computeType == TYPE_FP32) ? 1e-6f : 1e-3f;
    //float rtol  = (computeType == TYPE_FP32) ? 1e-4f : 1e-1f;
    bool  is_ok = false;
    if (sizeof(T) == 4) {
        is_ok = _checkResult<float>(name, out, ref, atol, rtol);
    }
    else {
        is_ok = _checkResult<half>(name, out, ref, atol, rtol);
    }
    return is_ok;
}

/// Compute Cosine Similarity

__global__ void convertHalfToFloat(float* dst, const half* src, const int size)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        dst[i] = __half2float(src[i]);
    }
}

template<typename T>
float CosineSimilarity(T* a, T* b, const int data_size)
{
    cublasHandle_t handle_cos;
    cublasCreate(&handle_cos);

    float *f_a = nullptr;
    float *f_b = nullptr;

    const bool IS_FP16   = std::is_same<T, half>::value;
    if(IS_FP16)
    {
        cudaMalloc((void**)(&f_a), sizeof(float)*data_size);
        cudaMalloc((void**)(&f_b), sizeof(float)*data_size);

        dim3 grid(32);
        dim3 block(256);
        convertHalfToFloat<<<grid, block, 0>>>(f_a, reinterpret_cast<half *>(a), data_size);
        convertHalfToFloat<<<grid, block, 0>>>(f_b, reinterpret_cast<half *>(b), data_size);
    }
    else
    {
        f_a = reinterpret_cast<float *>(a);
        f_b = reinterpret_cast<float *>(b);
    }

    float result=0.0, a_result=0.0, b_result=0.0;
   
    cublasSdot(handle_cos, data_size, f_a, 1, f_b, 1, &result);
    cublasSdot(handle_cos, data_size, f_a, 1, f_a, 1, &a_result);
    cublasSdot(handle_cos, data_size, f_b, 1, f_b, 1, &b_result);

    return result/max((sqrt(b_result)*sqrt(a_result)), 1e-8);
}

template float CosineSimilarity<half>(half* a, half* b, const int data_size);
template float CosineSimilarity<float>(float* a, float* b, const int data_size);

#define defineCheckResult(T, computeType)                                                            \
    template bool checkResult<T, computeType>(std::string name, TensorWrapper& out, TensorWrapper& ref, float atol, float rtol);

defineCheckResult(half, TYPE_FP32);
defineCheckResult(half, TYPE_FP16);


/// Analysis Times
void analysisTimes(const std::vector<float> times, float& mean, float& stdev)
{
    if(times.size() < 50)
    {
        printf("For accurate testing, Run kernel more times!\n");
        return;
    }

    std::vector<float>::const_iterator first = times.begin() + 40;
    std::vector<float>::const_iterator second = times.end();
    std::vector<float> real_time(first, second);

    double sum = std::accumulate(std::begin(real_time), std::end(real_time), 0.0);
    mean = sum / real_time.size();

    double accum = 0.0;
    std::for_each (std::begin(real_time), std::end(real_time), [&](const double d){
        accum += std::pow((d-mean), 2);
    });
    stdev = std::sqrt(accum/real_time.size());
}