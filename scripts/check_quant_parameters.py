################################################################################
# @Author   : zhen.wan@shopee.com
# @Date     : 2025-03-20 18:02:51
# @Details  : Output quantization parameters of weights (and activations) to files
################################################################################
import os
import sys
from lmdeploy.turbomind.deploy.loader import create_loader

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <model_path>")
    sys.exit(1)

model_path = sys.argv[1].rstrip('/')
output_dir = os.path.join(os.getcwd(), f"{os.path.basename(model_path)}_weight")
os.makedirs(output_dir, exist_ok=True)

loader = create_loader(model_path, r'model.layers.([0-9]+).')

try:
    for layer_idx, (layer_name, weights) in enumerate(loader.items()):
        output_path = os.path.join(output_dir, f"layers{layer_name}.txt")
        with open(output_path, 'w') as f:
            for k, v in weights.items():
                f.write(f'{k}: {v}, {v.shape}\n')
        print(f"Exported parameters of layers {layer_name} to {output_path}")

except Exception as e:
    print(f"Exported error: {e}")
