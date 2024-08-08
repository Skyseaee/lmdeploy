# Copyright (c) Shopee. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from lmdeploy.vl.model.onepiece.aip_logger import logger

def check_max_diff(a, b):
    diff = np.abs(a - b)
    return {"min diff": diff.min(), "mean diff": diff.mean(), "max diff": diff.max()}

def compute_cosine_similarity_numpy(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return {"cosine distance": np.dot(a, b.T) / (a_norm * b_norm + 1e-6)}

def compute_l1_distance(matrix1, matrix2):
    return torch.mean(torch.abs(matrix1 - matrix2)).item()

def compute_cosine_similarity(matrix1, matrix2):
    """
    Input[torch.Tensor]: matrix1, matrix2
    Return: scores
    """
    matrix1_flat = matrix1.view(1, -1)
    matrix2_flat = matrix2.view(1, -1)
    scores = F.cosine_similarity(matrix1_flat, matrix2_flat).item()
    return {"cosine distance": scores}

validate_map = {"cosine_distance": compute_cosine_similarity, "max_diff": check_max_diff}


def validate_output(validate_method, dict_a, dict_b):
    validate_func = validate_map[validate_method]
    for key in dict_a.keys():
        result = validate_func(dict_a[key], dict_b[key])
        logger.info(f"{result}, for output: {key}")
        logger.debug(f"torch result: \n{dict_a[key]}")
        logger.debug(f"trt result: \n{dict_b[key]}")

