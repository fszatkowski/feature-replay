#!/bin/bash

set -e

# Runs CIL experiments on Split MNIST from https://arxiv.org/abs/1904.07734

BENCHMARK="3s_split_mnist"
MEMORY_BUDGET=2000
CONSTANT_MEMORY=true

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Cumulative

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=["gdumb"] \
  strategy.memory_size=${MEMORY_BUDGET} \
  strategy.constant_memory=${CONSTANT_MEMORY}

python src/train.py \
  benchmark=${BENCHMARK} \
  strategy.base=Naive \
  strategy.plugins=["replay"] \
  strategy.memory_size=${MEMORY_BUDGET} \
  strategy.constant_memory=${CONSTANT_MEMORY}

for lwf_alpha in 0.01 0.05 0.1 0.5 1.0 1.5 2.0 5.0 10.0; do
  for lwf_temperature in 0.25 0.5 1.0 2.0 4.0; do
    python src/train.py \
      benchmark=${BENCHMARK} \
      strategy.base=Naive \
      strategy.plugins=["lwf"] \
      strategy.lwf_alpha=${lwf_alpha} \
      strategy.lwf_temperature=${lwf_temperature}
  done
done

for ewc_lambda in 0.01 0.05 0.1 0.5 1.0 1.5 2.0 5.0 10.0; do
  python src/train.py \
    benchmark=${BENCHMARK} \
    strategy.base=Naive \
    strategy.plugins=["ewc"] \
    strategy.ewc_lambda=${ewc_lambda}
done
