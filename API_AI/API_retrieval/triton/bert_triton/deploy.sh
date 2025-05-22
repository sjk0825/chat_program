#!/bin/bash

# too heavy. we replace tirton with huggingface inferenceapi. But triton also work
sudo docker run --rm --net=host \
  -v ./model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models