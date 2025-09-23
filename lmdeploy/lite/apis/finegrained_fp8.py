import os.path as osp
from pathlib import Path
from fire import Fire
from typing import List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    FineGrainedFP8Config,
    Qwen2_5_VLForConditionalGeneration,
)

from lmdeploy.archs import get_task


def finegrained_fp8(model: str,
                    work_dir: str = './work_dir',
                    ignored_layer_list: List[str] = ['lm_head'],
                    device: str = 'cuda',
                    revision: str = None,
                    download_dir: str = None):
    """
    Perform fine-grained fp8 quantization on both standard and MoE models.
    Credit to: https://huggingface.co/docs/transformers/quantization/finegrained_fp8

    Args:
        model (str): The path of model in hf format.
        work_dir (str): The working directory to save results.
        ignored_layers (`list[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `FP8Linear`.
    """

    if not osp.exists(model):
        print(f'can\'t find model from local_path {model}, '
              'try to download from remote')
        from lmdeploy.utils import get_model
        model = get_model(model, revision=revision, download_dir=download_dir)

    # Create work directory if not exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    _quant_config = {"modules_to_not_convert": ignored_layer_list, "quant_method": "fp8"}
    quant_config = FineGrainedFP8Config.from_dict(_quant_config)
    print(f"quant_config: {quant_config.to_dict()}")

    model_path = model
    model_type, _ = get_task(model_path)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              trust_remote_code=True)
    if model_type == 'llm':
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )
    elif model_type == 'vlm':
        model_arch = config.architectures[0]
        if model_arch == "Qwen2_5_VLForConditionalGeneration":
            quantized_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True
            )
        else:
            try:
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    quantization_config=quant_config,
                    trust_remote_code=True
                )
            except ValueError as e:
                raise ValueError(f"Fine-grained FP8 doesn't support {model_arch} now. ({e})")

    quantized_model.save_pretrained(work_dir)
    tokenizer.save_pretrained(work_dir)
    print(f"The model is saved to {work_dir}.")


if  __name__ == "__main__":
    Fire(finegrained_fp8)
