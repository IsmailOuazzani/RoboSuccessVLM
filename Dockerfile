FROM nvcr.io/nvidia/pytorch:24.03-py3

### Install InternVL Dependencies
RUN git clone https://github.com/OpenGVLab/InternVL.git \
    && cd InternVL \
    && git checkout 869d3be \
    && pip install -r requirements/internvl_chat.txt \
    && cd internvl_chat \
    && pip install .

# ImportError: cannot import name 'log' from 'torch.distributed.elastic.agent.server.api' (/home/ubuntu/venv/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py)
RUN pip install --upgrade deepspeed==0.14.4

# 500Gb RAM required for 64 jobs
# https://github.com/Dao-AILab/flash-attention/issues/1038#issuecomment-2439430999
RUN MAX_JOBS=40 pip install flash-attn==2.3.6 --no-build-isolation
RUN pip install datasets

RUN pip uninstall bitsandbytes -y \
    && git clone https://github.com/TimDettmers/bitsandbytes.git \
    && cd bitsandbytes \
    && cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCMAKE_BUILD_TYPE=Release  -S . -B /dev/shm/build \
    && cd .. \
    && cmake --build /dev/shm/build --parallel $(nproc)  \
    && cd bitsandbytes \
    && pip install .

# Cannot import name 'EncoderDecoderCache' from 'transformers'
RUN pip install peft==0.10.0
# AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
RUN pip install --upgrade opencv-python==4.8.0.74
# TypeError: Accelerator.__init__() got an unexpected keyword argument 'dispatch_batches'
RUN pip install "accelerate==0.34.2"


RUN pip freeze
