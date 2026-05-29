#! /usr/bin/env bash

export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="$PWD:$PYTHONPATH"

tvq_model_path="model"
dataset_path=s
output_path=a
datasets=dataset

mkdir -p $output_path

set -ue

python cli/llama_factory_cli.py train \
    ./configs/example.yaml \
    stage=sft \
    model_name_or_path=$tvq_model_path \
    dataset_dir=$dataset_path \
    output_dir=$output_path \
    dataset=$datasets 2>&1 | tee $output_path/log.txt