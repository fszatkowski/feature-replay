# Feature Replay in Continual Learning

## Setup
* Create environment with python 3.9
* Install requirements.txt
* [Setup Wandb](https://docs.wandb.ai/quickstart)
* Set [src](src) directory as PYTHONPATH

## Running experiments
Main script to run any experiment is [src/train.py](src/train.py). 
Please extend this script to add benchmarks, strategies, models.

We use hydra to manage experiments. Experiment parameters are stored in [config](config) directory. 

* To run training with default config, run the command below. 
Experiment parameters will be loaded from [config/config.yaml](config/config.yaml).
```shell
python src/train.py
```
* To override experiment parameters, use dot notation.
```shell
python src/train.py \
  seed=0 \
  model.hidden_sizes=[512,512,512] \
  training.optimizer.lr=0.0001
```
* To use load partial configs for benchmarks, strategies etc. use the notation from below.
Here *strategy* is the name of the [directory](config/strategy) in [config](config) and 
*basic_buffer* is the name of file in this directory (without .yaml extension).
```shell
python src/train.py \
  strategy=strategy.basic_buffer
```
