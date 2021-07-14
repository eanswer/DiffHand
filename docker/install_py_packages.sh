#!/bin/bash

set -euxo pipefail
pip install torch torchvision pyvista nevergrad
pip install numpy ninja imageio matplotlib joblib scipy