#!/usr/bin/env bash

# python3 a4cirrus/main.py softmax vit_base_patch16 imagenet


python3 a4cirrus/main.py \
--config /mnt/data/cx922/past/mase-tools/a4cirrus/experiments/quantile_int.yaml \
--model vit_base_patch16 --dataset imagenet