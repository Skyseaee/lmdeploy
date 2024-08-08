################################################################################
# @Copyright: 2019-2024 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2024-05-09 02:21:23
# @Details  : add model compress informations to config.yaml in model_path
################################################################################
from argparse import ArgumentParser
import yaml
import os
import logging


def main(model_path: str,
         model_format: str,
         output_model_path: str
         ):
    config_path = f"{model_path}/config.yaml"
    output_config_path = f"{output_model_path}/config.yaml"

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logging.warning(f"config.yaml dosen't exist in {model_path}, creating "
                        f"a default config.yaml in {output_config_path}.")
        config = {
            "modelinfo": {},
            "gunicorn": {
                "server_name": "0.0.0.0",
                "server_post": 80,
                "log_level": "ERROR",
            },
            "predictors": [
                {
                    "model_path": "./",
                    "cache-max-entry-count": 0.8
                }
            ],
        }

    version = os.getenv("ONELLM_VERSION", "0.6.3")
    config["modelinfo"]["model_format"] = 'fp8-w8a8' if 'fp8' == model_format else model_format
    config["modelinfo"]["framework"] = {
        "name": "onellm",
        "version": version,
    }

    with open(output_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path",
                        help="hf model path")
    parser.add_argument("--model_format",
                        help="model_format type: awq-w4a16, sq-int8, fp8")
    parser.add_argument("--output_model_path",
                        help="add model_format for config.yaml in output_model_path")
    args = parser.parse_args()

    main(**vars(args))
