import torch
try:
    import pycuda.driver as cuda
    import tensorrt as trt
    import numpy as np
except ModuleNotFoundError:
    raise RuntimeError(
        "You can't use TRTPredictor without tensorrt and "                                              # noqa: W503
        + "pycuda packages. You can get tensorrt by using a "                                           # noqa: W503
        + "tensorrt container from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt , "   # noqa: W503
        + "and get pycuda via pip install pycuda"                                                       # noqa: W503
    )

import packaging.version

from lmdeploy.utils import get_logger
logger = get_logger('lmdeploy')


def trt_version():
    return Version(trt.__version__)

class Version(packaging.version.Version):

    def __ge__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__ge__(other)

    def __le__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__le__(other)

    def __eq__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__eq__(other)

    def __gt__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__gt__(other)
    
    def __lt__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__lt__(other)

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif trt_version() >= '10.0' and dtype == trt.bfloat16:
        return torch.bfloat16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.fp8:
        return torch.float8_e4m3fn
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None, input_flattener=None, output_flattener=None):
        super(TRTModule, self).__init__()

        if isinstance(engine, str):
            # assume filepath
            with open(engine, 'rb') as f:
                engine = f.read()
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
        elif isinstance(engine, trt.IHostMemory):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
        self.engine = engine
        self.stream = cuda.Stream()  # 使用 pycuda.driver.Stream 创建非默认流
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
            self._update_name_binindgs_maps()
        self.input_names = input_names
        self.output_names = output_names
    
    def _update_name_binindgs_maps(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_io_tensors):
            name_i = self.engine.get_tensor_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _update_name_binding_maps_pre_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_bindings):
            name_i = self.engine.get_binding_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def forward(self, inputs_dicts, dtype=torch.float16):
        # 新增：确保输入数据在 GPU（使用自定义流异步拷贝）
        for input_name in self.input_names:
            tensor = inputs_dicts[input_name]
            if not tensor.is_cuda:
                # 使用 PyCUDA 的异步拷贝
                host_data = tensor.numpy()
                device_ptr = cuda.mem_alloc(tensor.nelement() * tensor.element_size())
                cuda.memcpy_htod_async(device_ptr, host_data, self.stream)
                inputs_dicts[input_name] = torch.as_tensor(
                    device_ptr, 
                    device="cuda", 
                    dtype=tensor.dtype
                ).view(tensor.shape)
        # set shapes
        for i, input_name in enumerate(self.input_names):
            indata = inputs_dicts[input_name]
            shape = indata.shape
            data_ptr = indata.contiguous().data_ptr()
            self.context.set_tensor_address(input_name, data_ptr)
            self.context.set_input_shape(input_name, shape)

        outputs = []
        for output_name in self.output_names:
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
            output = torch.empty(shape, dtype=dtype, device=device)
            self.context.set_tensor_address(output_name, output.data_ptr())
            outputs.append(output)

        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()

# Function to load the engine
def load_engine(engine_file_path, cfg, TRT_LOGGER = trt.Logger(trt.Logger.WARNING)):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    trtmodel = TRTModule(engine, cfg["input_names"], cfg["output_names"])
    return trtmodel
