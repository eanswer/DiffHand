#!/bin/bash

set -euxo pipefail
mkdir /workspace
cd /workspace
git clone https://github.com/eanswer/DiffHand.git --recursive -b docker
cd DiffHand
cd core
python setup.py install