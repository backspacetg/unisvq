#! /usr/bin/env bash

export PYTHONPATH="$PWD:$PYTHONPATH"

set -ue

base_model_path=/data/groups/QY_LLM_Other/wanghaoyu/pretrained_models/Qwen3-4B-instruct-2507
exp_dir=/data/groups/QY_LLM_Other/wanghaoyu/exp/quip_sharp/Qwen3-4B-instruct-2507
exp_name="Qwen3-4B-instruct-2507"
exp_suffix="_with_update"
dataset_path="/home/wanghaoyu2/data/subset_selection/pajama.jsonl"
stage=2

ckpt_path=${exp_dir}/ckpt${exp_suffix}
hf_path=${exp_dir}/hf${exp_suffix}
log_path=logs/${exp_name}${exp_suffix}

mkdir -p $log_path

if [[ $stage -eq 1 ]]; then

    python quantize_llama/quantize_with_update.py \
        --save_path $ckpt_path \
        --codebook linear_guassian \
        --batch_size 32 \
        --ft_bs 4 \
        --scale_override 0.83 \
        --base_model $base_model_path \
        --dataset_path $dataset_path \
        --devset_size 4096 \
        --ft_valid_size 128 \
        --scale_search_iters 1 \
        --ft_epoch 5 2>&1 | tee $log_path/log.txt
fi

if [[ $stage -eq 2 ]]; then
    mkdir -p $hf_path
    python quantize_llama/hfize.py \
        --base_model $base_model_path \
        --quantized_path $ckpt_path \
        --hf_output_path $hf_path
fi
