#! /usr/bin/bash
set -euxo pipefail

mkdir -p build && cd build

bash ../generate.sh
ninja -j$(nproc) && ninja install

cd .. && python3 -m pip install -e . --use-pep517
