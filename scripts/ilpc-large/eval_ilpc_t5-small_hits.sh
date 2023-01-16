#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-small
MODEL_TYPE=encoder-decoder
TASK_NAME=ilpc-large

TGT_LEN=512

METRIC=exact_match
SCHEDULER=linear
ITERS=250000
TBS=128
BS=128
MODEL_CFG="t5-small"

for SRC_LEN in 512
do
for LR in 5e-06
do
for N in 1
do
echo $MODEL_CFG
horovodrun --gloo -np $NP python run_finetuning_ilpc_hits.py \
        --validate_only \
        --task_name $TASK_NAME \
        --train_path /home/bulatov/bulatov/datasets/ilpc22/large_2sep_enum/large_train.csv \
        --valid_path /home/bulatov/bulatov/datasets/ilpc22/large_2sep_enum/large_valid.csv \
        --test_path /home/bulatov/bulatov/datasets/ilpc22/large_2sep_enum/large_test.csv \
        --model_path ./runs/test/t5-small/ilpc-large/lr5e-06_constant_with_warmup_adamw_wd1e-02_512-512_bs128_iters250000_pretrained_2sep_enum_eval/run_1  \
        --cpt_path ./runs/t5-small/ilpc-large/lr5e-06_constant_with_warmup_adamw_wd1e-02_512-512_bs128_iters250000_pretrained_2sep_enum/run_1 \
        --from_pretrained $MODEL_CFG \
        --tokenizer $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:T5ForConditionalGeneration \
        --use_generate_on_valid \
        --save_best \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/5)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/200)) --valid_interval $(($ITERS/50)) \
        --show_valid_examples 10 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42))
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"