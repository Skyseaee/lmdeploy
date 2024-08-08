# Copyright (c) Shopee. All rights reserved.
import os
import onnx

import torch
import time
from .onnx_helper import ONNXHelper
# from onnx_helper import ONNXHelper
from lmdeploy.vl.model.onepiece.aip_logger import logger

class TorchHelper:
    def __init__(self, model):
        self.model = model

    def export_onnx(
        self,
        onnx_file_path,
        input_names,
        input_shapes,
        output_names,
        input_dict,
        dynamic_axes=None,
        do_constant_folding=True,
        optimize=True,
        opset_version=16,
        export_params=True,
        verbose=False):        
        if not onnx_file_path:
            onnx_file_path = os.path.abspath(f"{str(time.time())}.onnx")
        else:
            saving_dir = os.path.abspath(os.path.split(onnx_file_path)[0])
            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)

        inputs = [input_dict[name].to("cuda") for name in input_names]
        self.model.to("cuda")
        torch.onnx.export(
            self.model,
            tuple(inputs),
            onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=export_params,
            verbose=verbose,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
        )
        if optimize:
            logger.info("🔔onnx model optimizing")
            try:
                onnx_helper = ONNXHelper(onnx_file_path)
                onnx_helper.optimize(
                    input_shapes=dict(zip(input_names, input_shapes)),
                    save_path=onnx_file_path,
                    dynamic_input_shape=dynamic_axes is not None,
                    overwrite=True,
                )
                onnx_helper.fold_constants(overwrite=True)
            except Exception:
                logger.warn("optimize onnx failed, do nothing")
        #load and check that onnx model is well-formed
        _model = onnx.load(onnx_file_path)
        try:
            onnx.checker.check_model(_model)
        except Exception as e:
            logger.error("The model is invalid: %s" % e)

        logger.info(f"onnx file Saved at: {onnx_file_path}")

        return onnx_file_path
