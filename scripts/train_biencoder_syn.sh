#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/biencoder_syn_$(date +%F-%H%M.%S)"
fi

#mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
#--train_file synret_50k.jsonl \
#--output_dir "${OUTPUT_DIR}" \
#CUDA_VISIBLE_DEVICES=0
# --add_prompts True \
# --q_prompt "query: " \
# --p_prompt "passage: " \
# python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_biencoder.py \
deepspeed ../src/train_biencoder.py --deepspeed ../ds_config.json \
    --model_name_or_path intfloat/multilingual-e5-base \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 256 \
	--gradient_accumulation_steps 2 \
    --add_pooler False \
    --t 0.02 \
    --seed 1234 \
    --do_train \
    --fp16 \
    --train_file syn_ret_nl.jsonl \
    --q_max_len 50 \
    --p_max_len 500 \
    --add_prompts True \
    --q_prompt "query: " \
    --p_prompt "passage: " \
	--train_tasks sl \
    --train_n_passages 16 \
    --use_in_batch_negs True \
    --full_contrastive_loss False \
    --dataloader_num_workers 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --use_scaled_loss True \
    --warmup_steps 200 \
    --share_encoder True \
    --logging_steps 50 \
    --save_total_limit 4 \
    --save_strategy steps \
    --save_steps 0.25 \
    --push_to_hub True \
    --hub_model_id Ehsanl/RetNLbase_sl_ibn16 \
    --hub_token \
    --eval_strategy steps \
    --eval_steps 0.25 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to wandb
