#!/bin/bash

set -e

for memory in 200 1000 6000 12000; do
  for ewc_lambda in 0.0 0.01 0.1 1.0 10. 100.; do
    hidden_size=300
    benchmark=split_fmnist
    epochs=5
    python src/analyze_feature_drift.py \
      benchmark=${benchmark} \
      strategy.memory_size=$memory \
      model.name="MLP" \
      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
      training.train_epochs=${epochs} \
      replay=false \
      ewc_lambda=$ewc_lambda \
      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}/model.pt"

    python src/analyze_feature_drift.py \
      benchmark=${benchmark} \
      strategy.memory_size=$memory \
      model.name="MLP" \
      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
      training.train_epochs=${epochs} \
      replay=true \
      ewc_lambda=$ewc_lambda \
      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}_replay/model.pt"
  done
done

for memory in 200 1000 5000; do
  for ewc_lambda in 0.0 0.01 0.1 1.0; do
    hidden_size=2000
    benchmark=cifar100
    epochs=20
    python src/analyze_feature_drift.py \
      benchmark=${benchmark} \
      strategy.memory_size=$memory \
      model.name="MLP" \
      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
      training.train_epochs=${epochs} \
      replay=false \
      ewc_lambda=$ewc_lambda \
      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}/model.pt"

    python src/analyze_feature_drift.py \
      benchmark=${benchmark} \
      strategy.memory_size=$memory \
      model.name="MLP" \
      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
      training.train_epochs=${epochs} \
      replay=true \
      ewc_lambda=$ewc_lambda \
      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}_replay/model.pt"
  done
done

rm -rf results/**/**/model.pt
