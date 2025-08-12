import os
import subprocess
from dataclasses import asdict
from load_server_testdata import load_config

def download_model(model_repo, target_dir):
    # modelrepo下载
    cmd = [
        "ais", "model", "download",
        "--model_id", str(model_repo.model_id),
        "--version_id", str(model_repo.version_id),
        "--output_path", target_dir,
        "--project", str(model_repo.project_id)
    ]

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if os.path.exists(os.path.join(target_dir, "config.json")):
        print("Download succeeded on first attempt")
        return

    ais_email = os.environ.get("AIS_LOGIN_EMAIL")
    ais_token = os.environ.get("AIS_LOGIN_TOKEN")
    print(f"Download failed, attempting to login with {ais_email}")
    if not ais_email or not ais_token:
        raise EnvironmentError("Missing environment variables: AIS_LOGIN_EMAIL or AIS_LOGIN_TOKEN")
    login_cmd = [
        "ais", "login",
        "--email", str(ais_email),
        "--token", str(ais_token),
        "--host", "https://ais.mlp.shopee.io"
    ]
    result_login = subprocess.run(login_cmd, text=True, timeout=30)
    if result_login.returncode != 0:
        raise RuntimeError(f"Modelrepo login failed: {result_login.stderr}")
    print("Retrying download after login...")
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")


def setup_models():
    print("Loading test info...")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(dir_path, "test_server_data.yaml"))
    import json
    print(json.dumps(asdict(config), indent=4))

    print("Downloading models if not exist...")
    models = []
    for _, model_config in config.test_models.items():
        model_path = model_config.model_info.model_path
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            print(f"Missing {config_file}, starting download...")
            download_model(model_config.model_info.model_repo, model_path)
        # verify
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Download failed: {config_file} not found")

        models.extend(list(model_config.service_info.items()))        

    return models

def pytest_configure(config):
    # 在pytest配置阶段下载模型（早于测试收集阶段）
    print("Setup  models...")
    models = setup_models()
    config.model_list = models
