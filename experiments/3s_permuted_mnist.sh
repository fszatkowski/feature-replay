#!/bin/bash

set -e

# Runs CIL experiments on Permuted MNIST from https://arxiv.org/abs/1904.07734

BENCHMARK="3s_permuted_mnist"
MEMORY_BUDGET=2000
CONSTANT_MEMORY=true
EWC_LAMBDA=50
LWF_ALPHA=1
LWF_TEMPERATURE=2
TAGS="[3s_permuted_mnist]"

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Cumulative \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=[gdumb] \
  strategy.memory_size=${MEMORY_BUDGET} \
  strategy.constant_memory=${CONSTANT_MEMORY} \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=[replay] \
  strategy.memory_size=${MEMORY_BUDGET} \
  strategy.constant_memory=${CONSTANT_MEMORY} \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=[lwf] \
  strategy.lwf_alpha=${LWF_ALPHA} \
  strategy.lwf_temperature=${LWF_TEMPERATURE} \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=[ewc] \
  strategy.EWC_LAMBDA=${EWC_LAMBDA} \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=[replay,lwf] \
  strategy.lwf_alpha=${LWF_ALPHA} \
  strategy.lwf_temperature=${LWF_TEMPERATURE} \
  strategy.memory_size=${MEMORY_BUDGET} \
  strategy.constant_memory=${CONSTANT_MEMORY} \
  wandb.tags=${TAGS}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=[replay,ewc] \
  strategy.EWC_LAMBDA=${EWC_LAMBDA} \
  strategy.memory_size=${MEMORY_BUDGET} \
  strategy.constant_memory=${CONSTANT_MEMORY} \
  wandb.tags=${TAGS}