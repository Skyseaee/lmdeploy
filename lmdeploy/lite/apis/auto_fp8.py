# Copyright (c) Shopee. All rights reserved.
import os.path as osp
from pathlib import Path
import shutil

import torch
from torch import nn
from transformers import AutoTokenizer

from lmdeploy.archs import get_task
from lmdeploy.lite.quantization.fp8 import (get_kv_cache_quant_layers,
                                            quantize_weights,
                                            quantize_activations)
from lmdeploy.lite.utils import get_calib_loaders, load_hf_from_pretrained

from .auto_awq import save_vl_model


def auto_fp8(model: str,
             work_dir: str = './work_dir',
             calib_dataset: str = 'ultrachat_2k',
             calib_samples: int = 128,
             batch_size: int = 1,
             calib_seqlen: int = 2048,
             act_scheme: str = 'static',
             kv_cache_fp8: bool = False,
             ignored_layer_list: list = ['lm_head'],
             device: str = 'cuda',
             revision: str = None,
             download_dir: str = None):
    """Perform fp8 quantization for using auto_fp8 algorithm.
    Credit to: https://github.com/neuralmagic/AutoFP8

    Args:
        model (str): The path of model in hf format.
        work_dir (str): The working directory to save results.
        calib_dataset (str): The calibration dataset name.
        calib_samples (int): The number of samples for calibration.
        batch_size (int): The batch size for running the calib samples.
            Low GPU mem requires small batch_size. Large batch_size
            reduces the calibration time while costs more VRAM.
        calib_seqlen (int): The maximum sequence length for calibration.
        act_scheme (str): Choice of either "dynamic" or "static" quantization.
            Default to static, If "static", then calibration samples are
            required during quantization to produce accurate per-tensor scales
            for activations of Linear modules.
        device (str): Device type of running.
        revision (str): The specific model version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified,
            will use the default version.
        download_dir (str): Directory to download and load the weights,
            default to the default cache directory of huggingface.
    """

    assert calib_dataset in ['c4', 'ptb', 'wikitext2','pileval',
                             'ultrachat_2k', 'llvm'], \
        'Support only `c4`, `ptb`, `wikitext2`, `pileval`, `ultrachat_2k` or \
        `llvm`.'

    # load model
    if not osp.exists(model):
        print(f'can\'t find model from local_path {model}, '
              'try to download from remote')
        from lmdeploy.utils import get_model
        model = get_model(model, revision=revision, download_dir=download_dir)

    model_path = model

    model_type, _ = get_task(model_path)
    if model_type == 'llm':
        # Load tokenizer and configuration
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  use_fast=False,
                                                  trust_remote_code=True)

        model = load_hf_from_pretrained(model_path,
                                        dtype="auto",
                                        device_map="auto",
                                        trust_remote_code=True)
        vl_model = None
    elif model_type == 'vlm':
        from lmdeploy.vl.model.builder import vl_model_with_tokenizer
        vl_model, model, tokenizer = vl_model_with_tokenizer(
            model_path=model_path)
        # fp8 calibrated on GPU
        model = model.to(device)


    kv_cache_quant_layers = []
    if kv_cache_fp8:
        kv_cache_quant_layers = get_kv_cache_quant_layers(model)
        if len(kv_cache_quant_layers) == 0:
            raise ValueError('Could not find any kv cache layers using '
                             'kv_cache_targets=["k_proj", "v_proj"], plsease '
                             'reset the kv_cache_targets.'
                             )

    # Always quantize the weights as they do not require calibration data
    print(f"ignored_layer_list: {ignored_layer_list}")
    quantize_weights(model, ignored_layer_list)

    if act_scheme == "static":
        kwargs = {'model_path': model_path}
        print(f'Loading calibrate {calib_dataset} dataset ...')
        calib_loader, _ = get_calib_loaders(calib_dataset,
                                            tokenizer,
                                            nsamples=calib_samples,
                                            seqlen=calib_seqlen,
                                            **kwargs)
        all_data = torch.cat([
            data if isinstance(data, torch.Tensor) else data[0]
            for data in calib_loader
        ]).to(device)

        # get quantization scales for activations
        quantize_activations(model, all_data, kv_cache_quant_layers,
                             ignored_layer_list)



    quantization_config = dict(quant_method='fp8',
                               version='gemm',
                               bits=8,
                               activation_scheme=act_scheme,
                               zero_point=False,
                               ignored_layers=ignored_layer_list,
                               kv_cache_scheme=(
                                   'fp8' if kv_cache_fp8 else None
                               ))
    model.config.update(dict(quantization_config=quantization_config))

    print(model)
    # Create work directory if not exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if vl_model:
        save_vl_model(vl_model, model_path, work_dir)
    else:
        model.save_pretrained(work_dir)
    tokenizer.save_pretrained(work_dir)
    print(f"The model is saved to {work_dir}")


if __name__ == '__main__':
    import fire
    fire.Fire(auto_fp8)
