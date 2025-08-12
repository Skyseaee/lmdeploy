# Copyright (c) Shopee. All rights reserved.
import os
from lmdeploy.vl.model.onepiece.details.torch_helper import TorchHelper
from lmdeploy.vl.model.onepiece.aip_logger import logger


def pt2onnx(
    torch_model,
    onnx_path,
    input_names,
    input_shapes,
    input_dict,
    output_names,
    dynamic_axes,
    opset_version=17,
    do_constant_folding=True,
    **kwargs,
):
    torch_model.eval()
    torch_helper = TorchHelper(torch_model)
    if os.path.exists(onnx_path):
        logger.warning(
            f"{onnx_path} already exists, skip convert. If you want to re-convert, please delete it first"
        )
        return onnx_path

    logger.info("begin to run torch -> onnx, this may take a few minutes.")
    onnx_path = torch_helper.export_onnx(
        onnx_path,
        input_names=input_names,
        input_shapes=input_shapes,
        input_dict=input_dict,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding
    )
    return onnx_path
