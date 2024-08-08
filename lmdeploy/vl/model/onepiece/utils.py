# Copyright (c) Shopee. All rights reserved.
import os
from typing import  Dict, Tuple, Union, List
import importlib.machinery
import importlib.metadata
import importlib.util
from functools import lru_cache

from packaging import version

import torch
from lmdeploy.vl.model.onepiece.aip_logger import logger

def gen_input_dict(input_names: List[str], input_shapes: List[List], input_dtypes: List[str], **kwargs) -> Dict:
    """Generate the input tensor to feed in model to test inference."""
    gen_shapes = input_shapes
    if 'max_input_shapes' in kwargs and kwargs['max_input_shapes']:
        gen_shapes = kwargs['max_input_shapes']
    input_dict = {}
    for name, shape, dtype in zip(input_names, gen_shapes, input_dtypes):
        tensor = torch.randn(shape, dtype=dtype)
        input_dict[name] = tensor

    return input_dict


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

@lru_cache()
def is_tensorrt_greater_or_equal(library_version: str="10.0"):
    if not _is_package_available("tensorrt"):
        return False

    return version.parse(importlib.metadata.version("tensorrt")) >= version.parse(library_version)

@lru_cache()
def is_transformers_greater_or_equal(library_version: str="4.49"):
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(library_version)

def is_support_optimize_vlm():
    # The current ImageEncoder model supports TensorRT as the backend only for CUDA version 12.0 and above. 
    # Otherwise, it uses the torch backend.
    # https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html#support-matrix
    if float(torch.version.cuda) >= 12.0 and torch.cuda.get_device_capability("cuda") >= (7, 5):
        ret = is_tensorrt_greater_or_equal("10.0")
        if not ret:
            logger.warn("try to prepare environment for tensorrt, about 1~10min")
            os.system(f"pip3 install -r {os.path.dirname(__file__)}/requirements.txt")
            ret = is_tensorrt_greater_or_equal("10.0")
        return ret
    else:
        logger.debug("😭This feature need condition: device capability >= 7.5")
        return False

def trt_version():
    _, version = _is_package_available("tensorrt", return_version=True)
    return version


def is_flash_attn_2_available(install_dependencies: bool = False) -> bool:
    import sys
    import os
    import torch
    if torch.cuda.get_device_capability("cuda") < (8, 0):
        return False

    _torch_available = _is_package_available("torch")
    if not _torch_available:
        return False

    if not _is_package_available("flash_attn"):
        if install_dependencies:
            os.system(f"{sys.executable} -m pip install flash_attn==2.8.0.post2")
            os.execv(sys.executable, [sys.executable] + sys.argv)
            if not _is_package_available("flash_attn"):
                return False
        else:
            return False

    # Let's add an extra check to see if cuda is available
    if not torch.cuda.is_available():
        return False
    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    # elif torch.version.hip:
    #     # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
    #     return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    else:
        return False
    

def device_default_half_type():
    if torch.cuda.get_device_capability("cuda") < (8, 0):
        return torch.half 
    else:
        return torch.bfloat16