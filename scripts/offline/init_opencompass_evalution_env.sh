# NOTE: 在主工程调用
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .

python3 -m pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ "aip-infer-utils>=0.6.2"

# Run in the OpenCompass directory
# cd opencompass
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip

