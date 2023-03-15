#!/bin/bash
export task=$1
declare -A lr=(["mrpc"]=5e-4 ["rte"]=1e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=1e-5 ["mnli"]=5e-4)
declare -A metrics=(["mrpc"]=accuracy ["rte"]=accuracy ["cola"]=matthews_correlation ["sst2"]=accuracy ["qnli"]=accuracy ["qqp"]=accuracy ["stsb"]=pearson ["mnli"]=accuracy)
declare -A eval_steps=(["mrpc"]=100 ["rte"]=100 ["cola"]=100 ["sst2"]=100 ["qnli"]=1000 ["qqp"]=1000 ["stsb"]=100 ["mnli"]=1000)

export sparsity=$2
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $task \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate ${lr[$task]} \
  --num_train_epochs 10 \
  --output_dir outs/fisher/${task}/${lr[$task]}/${sparsity} \
  --logging_steps 10 \
  --logging_dir logs/fisher/${task}/${lr[$task]}/${sparsity} \
  --evaluation_strategy steps \
  --eval_steps ${eval_steps[$task]} \
  --save_strategy steps \
  --save_steps ${eval_steps[$task]} \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --metric_for_best_model ${metrics[$task]} \
  --overwrite_output_dir \
  --initial_sparsity ${sparsity} \
  --num_samples 1024 \
  --cloze_task
