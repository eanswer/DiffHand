#!/bin/bash

set -euxo pipefail

add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils && \
     rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py
