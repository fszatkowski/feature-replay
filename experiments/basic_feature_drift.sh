#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

memory=$1

hidden_size=300
benchmark=split_fmnist
epochs=5

for ewc_lambda in 0.00 0.01 1.0 100.0; do
  python src/analyze_feature_drift.py \
    benchmark=${benchmark} \
    strategy.memory_size=$memory \
    model.name="MLP" \
    model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
    training.train_epochs=${epochs} \
    wandb.enable=false \
    replay=false \
    ewc_lambda=$ewc_lambda \
    output_dir="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}"

  python src/analyze_feature_drift.py \
    benchmark=${benchmark} \
    strategy.memory_size=$memory \
    model.name="MLP" \
    model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
    training.train_epochs=${epochs} \
    wandb.enable=false \
    replay=true \
    ewc_lambda=$ewc_lambda \
    output_dir="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}_replay"
done

for lwf_alpha in 0.1 1.0 10.0; do
  for lwf_temp in 0.5 1.0 2.0; do
    python src/analyze_feature_drift.py \
      benchmark=${benchmark} \
      strategy.memory_size=$memory \
      model.name="MLP" \
      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
      training.train_epochs=${epochs} \
      wandb.enable=false \
      replay=false \
      lwf_alpha=$lwf_alpha \
      lwf_temp=$lwf_temp \
      output_dir="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_lwf_alpha_${lwf_alpha}_temp_${lwf_temp}"

    python src/analyze_feature_drift.py \
      benchmark=${benchmark} \
      strategy.memory_size=$memory \
      model.name="MLP" \
      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
      training.train_epochs=${epochs} \
      wandb.enable=false \
      replay=true \
      lwf_alpha=$lwf_alpha \
      lwf_temp=$lwf_temp \
      output_dir="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_lwf_alpha_${lwf_alpha}_temp_${lwf_temp}_replay"
  done;
done



#for memory in 200 1000 6000 12000; do
#  for ewc_lambda in 0.0 0.01 0.1 1.0 10. 100.; do
#    hidden_size=300
#    benchmark=split_fmnist
#    epochs=5
#    python src/analyze_feature_drift.py \
#      benchmark=${benchmark} \
#      strategy.memory_size=$memory \
#      model.name="MLP" \
#      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
#      training.train_epochs=${epochs} \
#      replay=false \
#      ewc_lambda=$ewc_lambda \
#      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}/model.pt"
#
#    python src/analyze_feature_drift.py \
#      benchmark=${benchmark} \
#      strategy.memory_size=$memory \
#      model.name="MLP" \
#      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
#      training.train_epochs=${epochs} \
#      replay=true \
#      ewc_lambda=$ewc_lambda \
#      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}_replay/model.pt"
#  done
#done

#for memory in 200 1000 5000; do
#  for ewc_lambda in 0.0 0.01 0.1 1.0; do
#    hidden_size=2000
#    benchmark=cifar100
#    epochs=20
#    python src/analyze_feature_drift.py \
#      benchmark=${benchmark} \
#      strategy.memory_size=$memory \
#      model.name="MLP" \
#      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
#      training.train_epochs=${epochs} \
#      replay=false \
#      ewc_lambda=$ewc_lambda \
#      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}/model.pt"
#
#    python src/analyze_feature_drift.py \
#      benchmark=${benchmark} \
#      strategy.memory_size=$memory \
#      model.name="MLP" \
#      model.hidden_sizes=[${hidden_size},${hidden_size},${hidden_size}] \
#      training.train_epochs=${epochs} \
#      replay=true \
#      ewc_lambda=$ewc_lambda \
#      output_model_path="results/${benchmark}/mlp_3x${hidden_size}_${epochs}_epochs_mem_${memory}_ewc_${ewc_lambda}_replay/model.pt"
#  done
#done

rm -rf results/**/**/model.pt
