import os
import yaml
import torch
import importlib.metadata
import subprocess
import sys
from aipinfer import logger
from lmdeploy.cli.serve import SubCliServe


def set_env_vars_to_args(config_dict):
    for key in config_dict.keys():
        # 提供环境变量设置入口，以ONELLM_开头，大写和下划线
        env_var = f"ONELLM_{key.upper().replace('-', '_')}"
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if env_value.lower() == 'true':
                env_value = True
            elif env_value.lower() == 'false':
                env_value = False
            config_dict[key] = env_value
            # print(key, env_value, type(env_value))
            os.environ.pop(env_var)

def install_requirements(requirements: str):
    if requirements:
        packages = [pkg.strip() for pkg in requirements.split(',') if pkg.strip()]
        if packages:
            logger.info(f"Checking required packages: {packages}")
            to_install = []
            installed_packages = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()}
            for pkg in packages:
                pkg_name, _, pkg_version = pkg.partition("==")
                if pkg_name in installed_packages:
                    if not pkg_version or installed_packages[pkg_name] == pkg_version:
                        logger.info(f"Package {pkg} is already installed.")
                        continue
                to_install.append(pkg)

            if to_install:
                logger.info(f"Installing required packages: {to_install}")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install packages: {e}")
                    raise

def server_http(config_path: str):
    dir_path = os.path.dirname(os.path.realpath(config_path))
    config_dict = {}
    if config_path.endswith(".yaml"):
        config = yaml.safe_load(open(config_path).read())
    else:
        # use default parameters below
        config = {
            "modelinfo": {},
            "gunicorn": {},
            "predictors": [{}],
        }
    # load config from $AIP_ONELLM_MAAS_CONFIG
    maas_config_env = os.environ.get("AIP_ONELLM_MAAS_CONFIG")
    if maas_config_env:
        logger.info(f"Loading service params from AIP_ONELLM_MAAS_CONFIG: {maas_config_env}")
        try:
            yaml_string = maas_config_env.encode("utf-8").decode("unicode_escape")
        except Exception as e:
            yaml_string = maas_config_env
        config = yaml.safe_load(yaml_string)
        logger.info(f"Got service params from AIP_ONELLM_MAAS_CONFIG: {config}")
        if not isinstance(config, dict):
            raise ValueError(f"AIP_ONELLM_MAAS_CONFIG can not load as a dict, but got a {type(config)}")

    # Install requirements if specified
    install_requirements(os.environ.pop("ONELLM_REQUIREMENTS", config['predictors'][0].get("requirements", "")))

    model_format = config['modelinfo'].get("model_format", "fp16")
    model_format_env = os.environ.get("ONELLM_MODEL_FORMAT")
    if model_format_env:
        model_format = model_format_env
        os.environ.pop("ONELLM_MODEL_FORMAT")
    if model_format == "awq-w4a16":
        config_dict['model_format'] = 'awq'
    elif model_format == "fp8-w8a8":
        config_dict['quant_policy'] = 16  # default to fp8-kv8
        config_dict['model_format'] = 'fp8'

    config_dict["server_name"] = config['gunicorn'].get("server_name", "0.0.0.0")
    config_dict["server_port"] = config['gunicorn'].get("server_port", 80)
    config_dict["log_level"] = config['gunicorn'].get("log_level", "ERROR")
    # set config based on config.yaml of default values
    model_path = config['predictors'][0].get("model_path", "./")
    config_dict["cache_max_entry_count"] = config['predictors'][0].get("cache-max-entry-count", 0.8)
    config_dict["tp"] = config['predictors'][0].get("tp", torch.cuda.device_count())
    config_dict["max_batch_size"] = config['predictors'][0].get("max-batch-size", 128)
    config_dict["vision_max_batch_size"] = config['predictors'][0].get("vision-max-batch-size", 1)
    session_len = config['predictors'][0].get("session-len", None)
    max_concurrent_requests = config['predictors'][0].get("max-concurrent-requests", None)
    config_dict["enable_prefix_caching"] = config['predictors'][0].get("enable-prefix-caching", True)
    config_dict["enable_metrics"] = config['predictors'][0].get("enable-metrics", True)
    config_dict["max_log_len"] = config['predictors'][0].get("max-log-len", 1024)
    config_dict["model_name"] = config['predictors'][0].get("model-name", "")
    reasoning_parser = config['predictors'][0].get("reasoning-parser", None)
    tool_call_parser = config['predictors'][0].get("tool-call-parser", None)
    if 'quant_policy' in config['predictors'][0]:
        config_dict['quant_policy'] = config['predictors'][0]['quant_policy']
    elif 'quant-policy' in config['predictors'][0]:
        config_dict['quant_policy'] = config['predictors'][0]['quant-policy']
    else:
        config_dict['quant_policy'] = config_dict.get('quant_policy', 0)

    model_path = os.path.join(dir_path, model_path)

    # Process chat-template
    chat_template = config['predictors'][0].get("chat-template", None)
    chat_template_env = os.environ.get("ONELLM_CHAT_TEMPLATE")
    if chat_template_env:
        chat_template = chat_template_env
        os.environ.pop("ONELLM_CHAT_TEMPLATE")
    if chat_template:
        if os.path.isfile(os.path.join(dir_path, chat_template)):
            logger.info("The chat-template passed in is a valid file path")
            chat_template = os.path.join(dir_path, chat_template)
        config_dict["chat_template"] = chat_template

    # Env vars have higher priority
    set_env_vars_to_args(config_dict)

    # Pass args to lmdeploy SubCli interface
    SubCliServe.add_parsers()
    args_list = ["api_server", model_path]
    for key, value in config_dict.items():
        key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args_list.append(key)
        else:
            args_list.extend([key, str(value)])

    # some special args
    def handle_env_arg(env_var, arg_name, current_value):
        val = os.environ.pop(env_var, None)
        if val is not None:
            current_value = val
        if current_value:
            args_list.append(arg_name)
            args_list.append(str(current_value))
        return current_value
    handle_env_arg("ONELLM_SESSION_LEN",      "--session-len",      session_len)
    handle_env_arg("ONELLM_REASONING_PARSER", "--reasoning-parser", reasoning_parser)
    handle_env_arg("ONELLM_TOOL_CALL_PARSER", "--tool-call-parser", tool_call_parser)
    handle_env_arg("ONELLM_MAX_CONCURRENT_REQUESTS", "--max-concurrent-requests", max_concurrent_requests)

    server_args = SubCliServe.parser.parse_args(args_list)

    logger.info(f"Start http service with params: {server_args}")
    SubCliServe.api_server(server_args)

if __name__ == '__main__':
    import fire
    fire.Fire(server_http)
