#!/bin/bash
export task=$1
declare -A lr=(["mrpc"]=1e-5 ["rte"]=1e-4 ["cola"]=1e-5 ["sst2"]=1e-5 ["qnli"]=1e-4 ["qqp"]=1e-5 ["stsb"]=1e-5 ["mnli"]=2e-5 )
declare -A metrics=(["mrpc"]=accuracy ["rte"]=accuracy ["cola"]=matthews_correlation ["sst2"]=accuracy ["qnli"]=accuracy ["qqp"]=accuracy ["stsb"]=pearson ["mnli"]=accuracy )
declare -A eval_steps=(["mrpc"]=100 ["rte"]=100 ["cola"]=100 ["sst2"]=100 ["qnli"]=1000 ["qqp"]=1000 ["stsb"]=100 ["mnli"]=1000 )

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $task \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate ${lr[$task]} \
  --num_train_epochs 10 \
  --output_dir outs/mask_cloze/$task \
  --logging_steps 10 \
  --report_to tensorboard \
  --logging_dir logs/mask_cloze/$task \
  --evaluation_strategy steps \
  --eval_steps ${eval_steps[$task]} \
  --save_strategy steps \
  --save_steps ${eval_steps[$task]} \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model ${metrics[$task]} \
  --overwrite_output_dir \
  --initial_sparsity 0.05 \
  --cloze_task
