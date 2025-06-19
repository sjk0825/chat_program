#!/bin/bash

# too heavy. we replace tirton with huggingface inferenceapi. But triton also work
sudo docker run --rm --net=host \
  -v ./model_repository:/models \
  triton-python-transformers:latest \
  tritonserver --model-repository=/models