#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/biencoder_syn_$(date +%F-%H%M.%S)"
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
#--train_file synret_50k.jsonl \
#CUDA_VISIBLE_DEVICES=0
# python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_biencoder.py \
deepspeed ../src/train_biencoder.py --deepspeed ../ds_config.json \
    --model_name_or_path intfloat/multilingual-e5-large \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --add_pooler False \
    --t 0.02 \
    --seed 1234 \
    --do_train \
    --fp16 \
    --train_file syn_ret_nl.jsonl \
    --q_max_len 500 \
    --p_max_len 500 \
    --add_prompts True \
    --q_prompt "query: " \
    --p_prompt "passage: " \
    --train_n_passages 2 \
	--use_in_batch_negs False \
	--full_contrastive_loss False \
    --dataloader_num_workers 1 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --use_scaled_loss True \
    --warmup_steps 1000 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --save_total_limit 2 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
