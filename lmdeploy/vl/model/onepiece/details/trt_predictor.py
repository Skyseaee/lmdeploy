# Copyright (c) Shopee. All rights reserved.
try:
    from cuda import cuda, cudart
    import tensorrt as trt
except ModuleNotFoundError:
    raise RuntimeError(
        "You can't use TRTPredictor without tensorrt and "                                              # noqa: W503
        + "cuda-python packages. You can get tensorrt by using a "                                           # noqa: W503
        + "tensorrt container from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt , "   # noqa: W503
        + "and get pycuda via pip install cuda-python"                                                       # noqa: W503
    )

import torch

from lmdeploy.vl.model.onepiece.aip_logger import logger
def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call_res):
    err, res = call_res[0], call_res[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

def datatype_trt_to_torch(datatype_trt):
    # Cast TensorRT data type into Torch
    if datatype_trt == trt.float32:
        return torch.float32
    if datatype_trt == trt.float16:
        return torch.float16
    if datatype_trt == trt.int8:
        return torch.int8
    if datatype_trt == trt.int32:
        return torch.int32
    if datatype_trt == trt.bool:
        return torch.bool
    if datatype_trt == trt.uint8:
        return torch.uint8
    if datatype_trt == trt.DataType.FP8:
        return torch.float8_e4m3fn
    if datatype_trt == trt.bf16:
        return torch.bfloat16
    if datatype_trt == trt.int64:
        return torch.int64
    if datatype_trt == trt.int4:
        return None  # only torch.uint4 is supported
    return None

class TRTPredictor():
    def __init__(self, trt_engine_path, device):
        self._device = device
        self.model_path = trt_engine_path
        self._stream = cuda_call(cudart.cudaStreamCreate())
        self._engine = TRTPredictor.load_engine(self.model_path)
        self._context = self._engine.create_execution_context()

    def get_shapes(self):  # noqa: C901
        dynamic = False
        shape_dict = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._context.get_tensor_shape(name)
            if any([s < 0 for s in shape]):
                dynamic = True
                break

        if dynamic:
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    max_input_shape = self._engine.get_tensor_profile_shape(name, 0)[2]
                    self._context.set_input_shape(name, max_input_shape)

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._context.get_tensor_shape(name)
            shape_dict[name] = shape
        return shape_dict

    def __del__(self):
        del self._context
        del self._engine

    @staticmethod
    def load_engine(model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(model_path, "rb") as plan:
            return runtime.deserialize_cuda_engine(plan.read())

    def set_input(self, inputs_dict):
        for name, tensor in inputs_dict.items():
            shape = tensor.shape
            self._context.set_input_shape(name, shape)
            shape_dict = self.get_shapes()
            self._buffer = Buffer(self._engine, self._device, shape_dict)
            self._buffer.set_input(name, tensor)

    def do_inference_v3(self):
        for name, tensor in self._buffer.inputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())
        for name, tensor in self._buffer.outputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())
        self._context.execute_async_v3(stream_handle=self._stream)

    def get_output(self):
        outputs_dict = {}
        is_cuda = True if str(self._device).startswith('cuda') else False
        for name, out in self._buffer.outputs.items():
            if is_cuda:
                outputs_dict[name] = out
            else:
                outputs_dict[name] = out.cpu()
        return outputs_dict

    def __call__(self, inputs_dict):
        self.set_input(inputs_dict)
        self.do_inference_v3()
        outputs_dict = self.get_output()

        return outputs_dict


class Buffer(object):
    def __init__(self, engine,device,config=None):
        self._name2shape = config
        self._inputs = {}
        self._outputs = {}
        self._bindings = []
        self._device = device
        self._allocate_mem(engine)

    def _allocate_mem(self, engine):
        self.num_optimization_profiles = engine.num_optimization_profiles
        # print("----- engine num_optimization_profiles", self.num_optimization_profiles)
        self.num_bindings = engine.num_io_tensors
        self.num_binding_per_profile = (
            self.num_bindings // self.num_optimization_profiles
        )

        """
           self._bindings has multiple slots, but repeated slots uses the same device buffer ptr
        """
        # tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

        binding_idx = -1
        for name in engine:
            binding_idx += 1
            if binding_idx >= self.num_binding_per_profile:
                binding_ptr = self._bindings[binding_idx % self.num_binding_per_profile]
                self._bindings.append(binding_ptr)
                logger.debug(f"reuse {name}, {binding_idx}")
                continue
            # size = trt.volume(self._name2shape[name])
            dtype = engine.get_tensor_dtype(name)
            is_input = engine.get_tensor_mode(name=name) == trt.TensorIOMode.INPUT
            dtype = datatype_trt_to_torch(dtype)
            shape = tuple(self._name2shape[name])
            tensor = torch.empty(shape, dtype = dtype, device=self._device).contiguous()
            self._bindings.append(tensor)
            if is_input:
                self._inputs[name] = tensor
            else:
                self._outputs[name] = tensor

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def bindings(self):
        return self._bindings

    def set_input(self, name, tensor):
        try:
            if tensor.is_cuda:
                self._inputs[name] = tensor
            else:
                self._inputs[name] = tensor.to(device=self._device)
        except Exception as exc:
            raise exc
