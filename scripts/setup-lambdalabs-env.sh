#!/bin/bash
git clone git@github.com:OpenGVLab/InternVL.git
python3.10 -m venv venv
source venv/bin/activate

pip install wheel
pip install -r InternVL/requirements.txt
cd InternVL/internvl_chat && pip install . && cd ../..
pip install flash-attn==2.3.6 --no-build-isolation
pip install --upgrade deepspeed==0.14.4 # ImportError: cannot import name 'log' from 'torch.distributed.elastic.agent.server.api' (/home/ubuntu/venv/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py)
pip install datasets # ModuleNotFoundError: No module named 'datasets'

pip uninstall bitsandbytes
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
git checkout tags/0.44.0 # TODO: test, so far ive only done latest
cmake -DCOMPUTE_BACKEND=cuda -S . -B build
cmake --build build
pip install .
python -m bitsandbytes
cd ..

pip install "accelerate==0.34.2"
cd InternVL/internvl_chat
mkdir pretrained
