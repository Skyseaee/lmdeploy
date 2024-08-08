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

// fp8
#include "src/turbomind/utils/cublasFP8MMWrapper.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"

using namespace turbomind;
namespace ft = turbomind;

#define TIME_MS_START_TOOL(name, stream)                                                                               \
    cudaEvent_t _macro_event_start_##name, _macro_event_stop_##name;                                                   \
    cudaEventCreate(&_macro_event_start_##name);                                                                       \
    cudaEventCreate(&_macro_event_stop_##name);                                                                        \
    cudaEventRecord(_macro_event_start_##name, stream);

#define TIME_MS_END_TOOL(name, stream)                                                                                 \
    cudaEventRecord(_macro_event_stop_##name, stream);                                                                 \
    cudaEventSynchronize(_macro_event_stop_##name);                                                                    \
    float ms_##name = 0.0f;                                                                                            \
    cudaEventElapsedTime(&ms_##name, _macro_event_start_##name, _macro_event_stop_##name);                             \
    cudaEventDestroy(_macro_event_start_##name);                                                                       \
    cudaEventDestroy(_macro_event_stop_##name);                                                                        \
    // printf("[TIMEIT] " #name ": %.2fus\n", ms_##name*1000);

////////////////////////////////////////////////////////////////////////////////////

// TensorWrapper is to handle a tensor object as well as its memory buffer,
// because tensor.data is const we cannot set values.
class TensorWrapper {
private:
    IAllocator* allocator;

public:
    std::vector<size_t> shape;
    DataType            type;
    Tensor*             tensor;
    void*               data;

    TensorWrapper(IAllocator* allocator, DataType dtype, std::vector<size_t> shape, bool zero_init = false)
    {
        this->allocator = allocator;
        this->type      = dtype;
        this->shape     = shape;

        size_t tensor_memsize = this->memsize();
	this->data            = this->allocator->malloc(tensor_memsize, false);
        if (zero_init) {
            check_cuda_error(cudaMemset(data, 0x0, tensor_memsize));
        }
        else {
            setRandomValues();
        }
        this->tensor = new Tensor(MEMORY_GPU, dtype, shape, data);
    }

    TensorWrapper(TensorWrapper const& other):
        allocator(other.allocator), shape(other.shape), type(other.type), data(other.data), tensor(other.tensor)
    {
        TM_LOG_DEBUG("TensorWrapper copy: this=%p other=%p", data, other.data);
    }
    ~TensorWrapper()
    {
        delete tensor;
        allocator->free((void**)(&data));
    }

    void setInvalidValues()
    {
        size_t type_size   = tensor->type == TYPE_FP32 ? sizeof(float) : sizeof(half);
        size_t tensor_size = type_size * tensor->size();
        // Fill by a random number to guarantee invalid values
        check_cuda_error(cudaMemset(data, 0xdc, tensor_size));
    }

    void setRandomValues()
    {
        // random initialization
        size_t num_elements = this->size();
        switch (this->type) {
            case TYPE_FP32:
                cudaRandomUniform((float*)data, num_elements);
                break;
            case TYPE_FP16:
                cudaRandomUniform((half*)data, num_elements);
                break;
            case TYPE_FP8_E4M3:
                cudaRandomUniform((__nv_fp8_e4m3*)data, num_elements);
                break;
	    default:
                // Will be added more if needed.
                throw std::runtime_error("Not supported data type");
        }
    }

    size_t size()
    {
        size_t n_elements = 1;
        for (size_t s : this->shape) {
            n_elements *= s;
        }
        return n_elements;
    }

    size_t memsize()
    {
        size_t type_size = 0;
        switch (this->type) {
            case TYPE_FP32:
                type_size = sizeof(float);
                break;
            case TYPE_FP16:
                type_size = sizeof(half);
                break;
	    case TYPE_FP8_E4M3:
                type_size = sizeof(__nv_fp8_e4m3);
                break;
            default:
                throw std::runtime_error("Not supported data type.");
        }
        return type_size * this->size();
    }
};

template<typename T, DataType computeType>
bool checkResult(std::string name, TensorWrapper& out, TensorWrapper& ref, float atol, float rtol);

/// Compute Cosine Similarity
template<typename T>
float CosineSimilarity(T* a, T* b, const int data_size);

/// Compute performance in GFLOP/s
double gflops(const int m, const int n, const int k, const double runtime_s);

/// Analysis Times
void analysisTimes(const std::vector<float> times, float& mean, float& stdev);