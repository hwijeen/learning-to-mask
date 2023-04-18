#!/bin/bash
set -x
export src_task=$1
export tgt_task=$2
declare -A lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=1e-5 ["mnli"]=5e-4)
declare -A metrics=(["mrpc"]=accuracy ["rte"]=accuracy ["cola"]=matthews_correlation ["sst2"]=accuracy ["qnli"]=accuracy ["qqp"]=accuracy ["stsb"]=pearson ["mnli"]=accuracy)
# declare -A eval_steps=(["mrpc"]=100 ["rte"]=100 ["cola"]=100 ["sst2"]=100 ["qnli"]=1000 ["qqp"]=1000 ["stsb"]=100 ["mnli"]=1000)
export eval_steps=100


export sparsity=0.05
export lr=1e-5
export model_path=/projects/tir6/strubell/hahn2/mask/outs/mask_cloze/${src_task}/${lr[$src_task]}/0.05/42/emblin
# export model_path=bert-base-cased
python run_glue.py \
  --model_name_or_path $model_path \
  --dataset_name $tgt_task \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate $lr \
  --num_train_epochs 10 \
  --output_dir outs/mask_cloze_transfer/${src_task}_${tgt_task}/$lr/${sparsity} \
  --logging_steps 10 \
  --logging_dir logs/mask_cloze_transfer/${src_task}_${tgt_task}/$lr/${sparsity} \
  --evaluation_strategy steps \
  --eval_steps ${eval_steps[$tgt_task]} \
  --save_strategy steps \
  --save_steps ${eval_steps[$tgt_task]} \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --metric_for_best_model ${metrics[$tgt_task]} \
  --overwrite_output_dir \
  --initial_sparsity ${sparsity}\
  --cloze_task \
  # --max_train_samples 64
