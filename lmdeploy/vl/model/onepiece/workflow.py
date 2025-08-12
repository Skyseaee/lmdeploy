# Copyright (c) OpenMMLab. All rights reserved.
"""AIP Model Convert Workflow. We implement the use case in this file."""
import os
from typing import List, Dict

import torch
import time
from .aip_logger import logger
from .details.pt2onnx import pt2onnx
from .details.onnx2trt import onnx2trt
# from .details.trt_predictor import TRTPredictor
from .aip_validator import validate_output
from .details.trt_module import load_engine
from .utils import gen_input_dict

REPEAT_NUM = 100
def inference_torch(model, input_dict, device):
    model.to(device)
    ticks = []
    for _ in range(REPEAT_NUM):
        start = time.time()
        torch_input = (input_dict[k].to(device) for k in input_dict)
        outputs = model(*torch_input)
        end = time.time()
        tick = (end - start) * 1000
        ticks.append(tick)
    ticks = sorted(ticks)
    ticks = ticks[1:-1]
    if isinstance(outputs, torch.Tensor):
        # logger.info(f"torch output: {outputs.shape}")
        outputs = {'hidden_status': outputs.detach()}
    elif isinstance(outputs, dict):
        outputs = {k: v.detach() for k, v in outputs.items()}
    return outputs, sum(ticks) / len(ticks)

def inference_trt(trt, input_dict):
    ticks = []
    for _ in range(REPEAT_NUM):
        start = time.time()
        outputs = trt(input_dict)
        end = time.time()
        tick = (end - start) * 1000
        ticks.append(tick)
    ticks = sorted(ticks)
    ticks = ticks[1:-1]
    if isinstance(outputs, torch.Tensor):
        # logger.info(f"torch output: {outputs.shape}")
        outputs = {'hidden_status': outputs.detach()}
    elif isinstance(outputs, dict):
        outputs = {k: v.detach() for k, v in outputs.items()}
    return outputs, sum(ticks) / len(ticks)

def build_engine(cfg: Dict, model, device):
    if not os.path.exists(cfg["trt_path"]):
        pt2onnx(model, **cfg)
        trt_engine_path = onnx2trt(**cfg)
        trt = load_engine(trt_engine_path, cfg)
    else:
        logger.info(f"load engine from {cfg['trt_path']}")
        trt = load_engine(cfg["trt_path"], cfg)
    #
    # If cfg support input_dict, we will use the dummy data for inference check
    #
    input_dict = cfg.get('input_dict', None)
    if input_dict:
        for k, v in input_dict.items():
            input_dict[k] = v.cuda()
        ori_output, ori_cost = inference_torch(model, input_dict, device)
        test_output, test_cost = inference_trt(trt, input_dict)
        logger.debug(f"after optimization output {test_output}")
        validate_output(cfg["validate_method"], ori_output, test_output)
        if isinstance(ori_output, dict):
            for k in ori_output:
                logger.info(f"torch output: {k} {ori_output[k].shape} {ori_output[k]}")
                logger.info(f"opt output: {k} {test_output[k].shape} {test_output[k]}")
        logger.info(
            f"🔔[ONELLM] speed up {ori_cost / test_cost:.2f}, {ori_cost:.3f}ms vs {test_cost:.3f}ms"
        )
    return trt


def workflow(cfg: Dict, model):
    device = torch.device('cuda:0')
    trt = build_engine(cfg, model, device)
    out = trt(cfg['input_dict'])
    return out


def main():
    """Implement model convert pipeline.
         - cfg: the configuration for model convert.
    """
    cfg = {
        'onnx_path': 'siglip.onnx',  # the temp file when optimizing the model
        'trt_path': 'siglip.engine',  # the final output model you get

        # For multi inputs, the order of input_names, input_shapes and input_dtypes must be the same
        'input_names': ['pixel_value'],
        'input_shapes': [[1, 3, 384, 384]],
        'input_dtypes': [torch.float32],
        'output_names': ['img_emb'],

        # this is a dict to specify some dynamic axies of your input or output, 
        # for fixed input, set it to None
        'dynamic_axes': {
            'pixel_value':  {0: 'batch_size'},
            'img_emb': {0: 'batch_size'},
        },

        # this is used for tensorrt  dynamic inputs. if your input is fixed, just set them to None
        'min_input_shapes': [[1, 3, 384, 384]],
        'opt_input_shapes': [[2, 3, 384, 384]],
        'max_input_shapes': [[4, 3, 384, 384]],

        # options:[fp16, fp32], fp16 is much faster, but may have some consistency issues.
        'precision': 'fp16',
        'validate_method': 'cosine_distance',
    }
    if "input_dict" not in cfg:
        input_dict = gen_input_dict(**cfg)
        cfg["input_dict"] = input_dict
    else:
        input_dict = cfg["input_dict"]
    model_name = "google/siglip-so400m-patch14-384"
    device = torch.device('cuda:0')
    from transformers import SiglipVisionModel
    model = SiglipVisionModel.from_pretrained(model_name).to(device)
    model = model.eval()
    trt = build_engine(cfg, model, device)
    ori_output, ori_cost = inference_torch(model, input_dict, device)
    test_output, test_cost = inference_trt(trt, input_dict)
    logger.debug(f"after optimization output {test_output}")
    validate_output(cfg["validate_method"], ori_output, test_output)
    logger.info(
        f"speed up {ori_cost / test_cost:.2f}, {ori_cost:.3f}ms vs {test_cost:.3f}ms"
    )
    logger.info("finish")


if __name__ == "__main__":
    main()
