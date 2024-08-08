# Copyright (c) Shopee. All rights reserved.
import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa:
from packaging.version import Version

from lmdeploy.vl.model.onepiece.details.utils import check_and_raise

from tensorrt import IConvolutionLayer,           \
    IActivationLayer, IPoolingLayer, ILRNLayer, IScaleLayer, ISoftMaxLayer,      \
    IConcatenationLayer, IDeconvolutionLayer, IElementWiseLayer, IGatherLayer,   \
    IPluginV2Layer, IUnaryLayer, IReduceLayer, IPaddingLayer,                    \
    IParametricReLULayer, ISelectLayer, IShuffleLayer, ISliceLayer, IShapeLayer, \
    ITopKLayer, IMatrixMultiplyLayer, IRaggedSoftMaxLayer, IIdentityLayer,       \
    IConstantLayer, IResizeLayer, IFillLayer, IQuantizeLayer, IDequantizeLayer,  \
    IScatterLayer, IConditionLayer, IIfConditionalOutputLayer,                   \
    IIfConditionalInputLayer, IEinsumLayer, IAssertionLayer, ILayer
from collections import defaultdict
from lmdeploy.vl.model.onepiece.aip_logger import logger, TRT_LOG_LEVEL

LOG_FUNC_MAP = {
    trt.ILogger.VERBOSE: logger.debug,
    trt.ILogger.INFO: logger.info,
    trt.ILogger.WARNING: logger.warning,
    trt.ILogger.ERROR: logger.error,
    trt.ILogger.INTERNAL_ERROR: logger.exception,
}

LOG_LEVEL_MAP = {
    "TRACE": trt.ILogger.VERBOSE,
    "DEBUG": trt.ILogger.VERBOSE,
    "INFO": trt.ILogger.INFO,
    "SUCCESS": trt.ILogger.INFO,
    "WARNING": trt.ILogger.WARNING,
    "ERROR": trt.ILogger.ERROR,
    "CRITICAL": trt.ILogger.INTERNAL_ERROR,
}

LAYER_DICT = {
    "LayerType.CONVOLUTION": IConvolutionLayer,
    "LayerType.ACTIVATION": IActivationLayer,
    "LayerType.POOLING": IPoolingLayer,
    "LayerType.LRN": ILRNLayer,
    "LayerType.IScaleLayer": IScaleLayer,
    "LayerType.SOFTMAX": ISoftMaxLayer,
    "LayerType.CONCATENATION": IConcatenationLayer,
    "LayerType.DECONVOLUTION": IDeconvolutionLayer,
    "LayerType.ELEMENTWISE": IElementWiseLayer,
    "LayerType.GATHER": IGatherLayer,
    "LayerType.PLUGIN_V2": IPluginV2Layer,
    "LayerType.UNARY": IUnaryLayer,
    "LayerType.REDUCE": IReduceLayer,
    "LayerType.PADDING": IPaddingLayer,
    "LayerType.PARAMETRIC_RELU": IParametricReLULayer,
    "LayerType.SELECT": ISelectLayer,
    "LayerType.SHUFFLE": IShuffleLayer,
    "LayerType.SLICE": ISliceLayer,
    "LayerType.SHAPE": IShapeLayer,
    "LayerType.TOPK": ITopKLayer,
    "LayerType.MATRIX_MULTIPLY": IMatrixMultiplyLayer,
    "LayerType.RAGGED_SOFTMAX": IRaggedSoftMaxLayer,
    "LayerType.IDENTITY": IIdentityLayer,
    "LayerType.CONSTANT": IConstantLayer,
    "LayerType.RESIZE": IResizeLayer,
    "LayerType.FILL": IFillLayer,
    "LayerType.QUANTIZE": IQuantizeLayer,
    "LayerType.DEQUANTIZE": IDequantizeLayer,
    "LayerType.SCATTER": IScatterLayer,
    "LayerType.CONDITION": IConditionLayer,
    "LayerType.CONDITIONAL_OUTPUT": IIfConditionalOutputLayer,
    "LayerType.CONDITIONAL_INPUT": IIfConditionalInputLayer,
    "LayerType.EINSUM": IEinsumLayer,
    "LayerType.ASSERTION": IAssertionLayer,
}


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """Implements the INT8 Entropy Calibrator 2."""

    def __init__(self, cache_file):
        """:param cache_file: The location of the cache file."""
        super().__init__()
        self.cache_file = cache_file
        self.data_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_data_batcher(self, data_batcher):
        """Define the image batcher to use, if any. If using only the cache file, \
        an image batcher doesn't need to be defined.

        :param data_batcher: The data object
        """
        # self._cuda_ctx.push()
        self.data_batcher = data_batcher
        size = int(np.dtype(self.data_batcher.dtype).itemsize * np.prod(self.data_batcher.shape))

        logger.debug("size: {size}")

        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.data_batcher.get_batch()

    def get_batch_size(self):
        """Get the batch size to use for calibration.

        :return: Batch size.
        """
        if self.data_batcher:
            return self.data_batcher.batch_size
        return 1

    def get_batch(self, names):
        """Get the next batch to use for calibration, as a list of device memory pointers.

        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.data_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            logger.debug(
                f"Calibrating image {self.data_batcher.image_index} / {self.data_batcher.num_images}"
            )
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            logger.debug("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """Read the calibration cache file stored on disk, if it exists.

        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.warning(
                    f"Using calibration cache file: {os.path.abspath(self.cache_file)}"
                )
                return f.read()

    def write_calibration_cache(self, cache):
        """Store the calibration cache to a file on disk.

        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            logger.info(
                f"Writing calibration cache data to: {os.path.abspath(self.cache_file)}"
            )
            f.write(cache)


class CustomLogger(trt.ILogger):
    def __init__(self, level):
        trt.ILogger.__init__(self)
        self._level = level

    def log(self, severity, msg):
        if severity > self._level:
            return
        log_func = LOG_FUNC_MAP[severity]
        log_func(msg)


class EngineBuilder:
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(self, max_workspace_size):
        trt_log_level = LOG_LEVEL_MAP[TRT_LOG_LEVEL]
        self.trt_logger = CustomLogger(trt_log_level)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        if Version(trt.__version__) >= Version("8.6.1"):
            self.config.builder_optimization_level = \
                int(os.environ.get("AIP_TRT_BUILDER_OPTIMIZATION_LEVEL", 3))
        if Version(trt.__version__) < Version("8.4"):
            self.config.max_workspace_size = max_workspace_size
        else:
            self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        self.network = None
        self.parser = None
        logger.info(f"😊TensorRT Version: {trt.__version__}")

    def getattr_recursive(self, source, attr_str):
        if attr_str is None:
            return None
        attrs = attr_str.split(".")
        ret = source
        for attr in attrs:
            ret = getattr(ret, attr)
        return ret

    def create_network(self, onnx_path):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                logger.error(f"Failed to load ONNX file: {onnx_path}")
                for error in range(self.parser.num_errors):
                    logger.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        logger.info("Network Description")
        for input in inputs:
            logger.info(
                f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}"
            )
        for output in outputs:
            logger.info(
                f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}"
            )

    def set_precision_patterns(self, precision_patterns):
        if precision_patterns is None:
            return
        for layer_index in range(self.network.num_layers - 1):
            layer: ILayer = self.network.get_layer(layer_index)
            next_layer: ILayer = self.network.get_layer(layer_index + 1)
            # POW operation usually followed by mean reduce

            for p in precision_patterns:
                layer_type = self.getattr_recursive(trt, p["layer.type"])
                next_layer_type = self.getattr_recursive(trt, p["next_layer.type"])

                layer_op = self.getattr_recursive(trt, p["layer.op"])
                next_layer_op = self.getattr_recursive(trt, p["next_layer.op"])

                layer_child_type = self.getattr_recursive(trt, p["layer.child_type"])
                next_layer_child_type = self.getattr_recursive(
                    trt, p["next_layer.child_type"]
                )

                target_precision = self.getattr_recursive(trt, p["target_precision"])

                if layer_type in [None, layer.type] and next_layer_type in [
                    None,
                    next_layer.type,
                ]:
                    # casting to get access to op attribute

                    # ---------- backup the __class__ ----------
                    tmp_layer_class = layer.__class__
                    tmp_next_layer_class = next_layer.__class__
                    # ------------------------------------------

                    layer.__class__ = LAYER_DICT[str(layer.type)]
                    next_layer.__class__ = LAYER_DICT[str(next_layer.type)]
                    if (layer_op in [None, layer.op if hasattr(layer, "op") else None]
                        and next_layer_op in [None, next_layer.op if hasattr(next_layer, "op") else None]  # noqa: W503
                        and layer_child_type in [None, layer.type if hasattr(layer, "type") else None]     # noqa: W503
                        and next_layer_child_type                                                          # noqa: W503
                            in [None, next_layer.type if hasattr(next_layer, "type") else None]):          # noqa: W503

                        layer.precision = target_precision
                        next_layer.precision = target_precision

                        for output_idx in range(layer.num_outputs):
                            layer.set_output_type(
                                index=output_idx, dtype=target_precision
                            )
                        logger.info(
                            f"set layer {layer.name} precision to {target_precision} ..."
                        )

                        if (
                            next_layer_op is not None or next_layer_type is not None
                        ):  # need to involve next layer
                            for output_idx in range(next_layer.num_outputs):
                                next_layer.set_output_type(
                                    index=output_idx, dtype=target_precision
                                )

                            logger.info(
                                f"set next_layer_op {next_layer.name} precision to {target_precision} ..."
                            )

                    # ---------- recovery the __class__ ----------
                    layer.__class__ = tmp_layer_class
                    next_layer.__class__ = tmp_next_layer_class
                    # --------------------------------------------

    def create_engine(    # noqa: C901
        self,
        engine_path,
        precision,
        calib_cache=None,
        data_batcher=None,
        input_names=["input"],
        minimum_input_shapes=None,
        optimization_input_shapes=None,
        maximum_input_shapes=None,
        precision_patterns=None,
        enable_preview_feature=False,
        timing_cache=None,
        *args,
        **kwargs
    ):
        """Build the TensorRT engine and serialize it to disk.

        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param data_batcher: The data batcher including calib data.
        :param minimum_input_shapes: Optimizated mininum input shape before trt8.6.
        :param optimization_input_shapes: Optimizated optimization input shape before trt8.6.
        :param maximum_input_shapes: Optimizated maximum input shape before trt8.6.
        :param precision_patterns: .
        :param enable_preview_feature: trt8.5 new feature.
        """
        check_and_raise(ValueError, precision in ["fp32", "fp16", "int8"], "")

        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        logger.info(f"Building {precision} Engine in {engine_path}")

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)

                # TODO: --------------- remove this
                if os.path.exists(calib_cache):
                    os.remove(calib_cache)
                # TODO: ---------------------------

                if not os.path.exists(calib_cache):
                    check_and_raise(ValueError, data_batcher is not None, "")
                    self.config.int8_calibrator.set_data_batcher(data_batcher)

        # ------------------ analyze ------------------
        analyze_dict = defaultdict(int)
        for layer_index in range(self.network.num_layers):
            layer: ILayer = self.network.get_layer(layer_index)

            k = str(layer.type)
            if hasattr(layer, "op"):
                k = f"{k}__{layer.op}"
            if hasattr(layer, "type"):
                k = f"{k}__{layer.type}"
            analyze_dict[k] += 1

        for k in analyze_dict:
            logger.info(f"{k}: {analyze_dict[k]}")

        if precision == "fp16":
            self.set_precision_patterns(precision_patterns)
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        #
        # Optimize runtime dimensions with TensorRT’s DL Compiler.
        # Potentially reduces run time and decreases device memory usage and engine size.
        # Models most likely to benefit from enabling FASTER_DYNAMIC_SHAPES_0805 are
        # transformer-based models, and models containing dynamic control flows.
        #
        # if Version(trt.__version__) >= Version("8.5"):
        #     logger.info(f"preview_feature: {enable_preview_feature}")
        #     self.config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805,
        #                                     enable=enable_preview_feature)
        # load global timing cache
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#timing-cache
        logger.info(f"timing_cache: {timing_cache is not None}")
        if Version(trt.__version__).major >= 8 and timing_cache is not None:
            if os.path.exists(timing_cache):
                with open(timing_cache, "rb") as f:
                    cache = self.config.create_timing_cache(f.read())
                    self.config.set_timing_cache(cache, ignore_mismatch=False)
            else:
                cache = self.config.create_timing_cache(b"")
                self.config.set_timing_cache(cache, ignore_mismatch=False)

        is_dynamic = False
        if optimization_input_shapes:
            is_dynamic = True
            check_and_raise(
                ValueError, minimum_input_shapes and maximum_input_shapes, ""
            )
        logger.info(f"is_dynamic: {is_dynamic}")

        if is_dynamic:
            check_and_raise(ValueError, optimization_input_shapes is not None, "")
            check_and_raise(ValueError, maximum_input_shapes is not None, "")
            check_and_raise(ValueError, minimum_input_shapes is not None, "")
            check_and_raise(ValueError, len(input_names) == len(minimum_input_shapes) == len(maximum_input_shapes), "")

            profile = self.builder.create_optimization_profile()

            logger.info(f"input_names: {input_names}")
            logger.info(f"minimum_input_shapes: {minimum_input_shapes}")
            logger.info(f"optimization_input_shapes: {optimization_input_shapes}")
            logger.info(f"maximum_input_shapes: {maximum_input_shapes}")

            if precision == "int8":
                for input_name, minimum_shape, opt_shape, maximum_shape in zip(
                    input_names,
                    minimum_input_shapes,
                    optimization_input_shapes,
                    maximum_input_shapes,
                ):
                    profile.set_shape(
                        input_name,
                        [1] + minimum_shape[1:],
                        [1] + opt_shape[1:],
                        [1] + maximum_shape[1:],
                    )
                self.config.set_calibration_profile(profile)

            else:
                for input_name, minimum_shape, opt_shape, maximum_shape in zip(
                    input_names,
                    minimum_input_shapes,
                    optimization_input_shapes,
                    maximum_input_shapes,
                ):
                    profile.set_shape(
                        input_name, minimum_shape, opt_shape, maximum_shape
                    )

            self.config.add_optimization_profile(profile)

        with open(engine_path, "wb") as f:
            with self.builder.build_serialized_network(self.network, self.config) as serialize_model:
                logger.info(f"Serializing {precision} engine to file: {engine_path}")
                f.write(serialize_model)

        # save timing_cache
        if Version(trt.__version__).major >= 8 and timing_cache is not None:
            cache = self.config.get_timing_cache()
            with cache.serialize() as buffer, open(timing_cache, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f)
