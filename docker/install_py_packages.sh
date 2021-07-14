#!/bin/bash

set -euxo pipefail
pip install torch==1.7.1 torchvision==0.8.2 pyvista==0.31.3 nevergrad==0.4.3
pip install numpy==1.20.2 ninja==1.10.0.post2 imageio==2.9.0 matplotlib==3.4.2 joblib==1.0.1 scipy==1.5.2