#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=encoder-decoder
TASK_NAME=contract_nli

SRC_LEN=512
TGT_LEN=512

METRIC=exact_match
SCHEDULER=linear
ITERS=3000000
TBS=32
BS=8

MODEL_NAME=t5-small
LR=2e-04
N=1

python run_finetuning_scrolls.py \
        --task_name $TASK_NAME \
        --model_path ./runs/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${SRC_LEN}-${TGT_LEN}_bs${TBS}_iters${ITERS}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:T5ForConditionalGeneration \
        --use_generate_on_valid \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/40)) --valid_interval $(($ITERS/40)) \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42))
echo "done"
