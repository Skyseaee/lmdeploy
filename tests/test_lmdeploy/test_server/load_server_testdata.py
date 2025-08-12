import os
import copy
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class ModelRepo:
    model_id: int
    version_id: int
    project_id: int = 0


@dataclass
class TestData:
    prompt: str
    ground_truth: str
    verification: str
    image_url: Optional[str] = None


@dataclass
class ServiceInfo:
    model_path: str
    service_args: str
    test_data: List[TestData]
    quantization: Optional[str] = None
    target_gpus:List[str] = None


@dataclass
class ModelInfo:
    model_path: str
    model_repo: Optional[ModelRepo] = None

@dataclass
class TestModel:
    model_info: ModelInfo
    service_info: Dict[str, ServiceInfo]


@dataclass
class ServerTestConfig:
    common_settings: Dict[str, str]
    test_models: Dict[str, TestModel]

# 直接在测试config准备阶段，对GPU进行过滤，只取对应GPU的内容
def check_gpu():
    is_hopper = False
    gpu_name = ""
    import torch
    if not torch.cuda.is_available():
        return is_hopper, gpu_name
    device = torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    if cc_major == 9:
        is_hopper = True

    gpu_lists = {"v100", "t4", "a30", "a100", "h100"}
    gpu_name = torch.cuda.get_device_name(device).lower()
    for keyword in gpu_lists:
        if keyword in gpu_name:
            return is_hopper, keyword
    return is_hopper, gpu_name

def load_config(file_path: str) -> ServerTestConfig:
    with open(file_path, "r") as f:
        raw_data = yaml.safe_load(f)

    is_hopper, gpu_name = check_gpu()


    common_settings = raw_data["common_settings"]
    default_model_dir = common_settings["model_dir"]
    test_models = {}

    for model_name, model_data in raw_data["test_models"].items():
        model_dir = model_data["model_info"].get("model_dir", default_model_dir)
        model_path = os.path.join(model_dir, model_name)

        model_repo_data = model_data["model_info"].get("model_repo")
        model_repo = ModelRepo(**model_repo_data) if model_repo_data else None

        model_info = ModelInfo(model_path=model_path, model_repo=model_repo)

        common_service_args = ""
        common_service_args_dict = model_data["service_info"]["common"].get("service_args", {})
        if common_service_args_dict:
            common_service_args = common_service_args_dict.get("common", "")
            common_service_args = f'{common_service_args} {common_service_args_dict.get(gpu_name, "")}'.strip()
        common_test_data = [
            TestData(**td) for td in model_data["service_info"]["common"].get("test_data", [])
        ]

        service_info = {}
        for service_name, service_data in model_data["service_info"].items():
            if service_name == "common":
                continue
            if gpu_name not in service_data.get("target_gpus", []):
                continue
            service_args = ""
            service_args_dict = service_data.get("service_args", {})
            if service_args_dict:
                service_args = service_args_dict.get("common", "")
                service_args = f'{service_args} {service_args_dict.get(gpu_name, "")}'.strip()

            full_service_args = f"{common_service_args} {service_args}".strip()

            test_data = [
                TestData(**td) for td in service_data.get("test_data", [])
            ] + common_test_data
            # 过滤下GPU
            test_data = [copy.deepcopy(data) for data in test_data if gpu_name in data.ground_truth]
            for data in test_data:
                if gpu_name in data.ground_truth:
                    data.ground_truth = data.ground_truth[gpu_name]
            if not test_data:
                continue

            service_info[f"{model_name}-{service_name}".strip()] = ServiceInfo(
                model_path=model_path,
                service_args=full_service_args,
                test_data=test_data,
                quantization=service_data.get("quantization"),
            )
        if service_info:
            test_models[model_name] = TestModel(model_info=model_info, service_info=service_info)

    return ServerTestConfig(common_settings=common_settings, test_models=test_models)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(dir_path, "test_server_data.yaml"))
    import json
    print(json.dumps(asdict(config), indent=4))
