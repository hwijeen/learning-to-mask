#!/bin/bash
export task=$1
declare -A bz=( ["mrpc"]=32 ["rte"]=32 ["cola"]=32 ["sst2"]=128 ["qnli"]=128 )
declare -A lr=(["mrpc"]=1e-5 ["rte"]=1e-5 ["cola"]=1e-5 ["sst2"]=1e-5 ["qnli"]=7e-4 )

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $task \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size ${bz[$task]} \
  --learning_rate ${lr[$task]} \
  --num_train_epochs 10 \
  --output_dir outs/$task \
  --logging_steps 10 \
  --report_to tensorboard \
  --logging_dir logs/$task \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy no \
  --overwrite_output_dir \
  --initial_sparsity 0.1 \
  --cloze_task