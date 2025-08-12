# Copyright (c) Shopee. All rights reserved.
import os
from lmdeploy.vl.model.onepiece.aip_logger import logger
from lmdeploy.vl.model.onepiece.details.onnx_helper import ONNXHelper

def onnx2trt(
    onnx_path,
    trt_path,
    input_names,
    output_names,
    min_input_shapes,
    opt_input_shapes,
    max_input_shapes,
    precision,
    max_workspace_size=15 * (2**30),  # 15G
    **kwargs,
):
    if os.path.exists(trt_path):
        logger.warning(
            f"{trt_path} already exists, skip convert. If you want to re-convert, please delete it first"
        )
        return trt_path
    logger.info("begin to run onnx -> tensorrt, this may take minutes to hours..")

    onnx_helper = ONNXHelper(onnx_path)
    onnx_helper.export_trt_engine(
        trt_path,
        input_names=input_names,
        minimum_input_shapes=min_input_shapes,
        optimization_input_shapes=opt_input_shapes,
        maximum_input_shapes=max_input_shapes,
        precision=precision,
        max_workspace_size=max_workspace_size,
        **kwargs
    )
    return trt_path
