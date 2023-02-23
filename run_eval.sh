#!/bin/bash
export task=$1
declare -A lr=(["mrpc"]=5e-4 ["rte"]=1e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=1e-5 ["mnli"]=5e-4)
declare -A metrics=(["mrpc"]=accuracy ["rte"]=accuracy ["cola"]=matthews_correlation ["sst2"]=accuracy ["qnli"]=accuracy ["qqp"]=accuracy ["stsb"]=pearson ["mnli"]=accuracy)
declare -A eval_steps=(["mrpc"]=100 ["rte"]=100 ["cola"]=100 ["sst2"]=100 ["qnli"]=1000 ["qqp"]=1000 ["stsb"]=100 ["mnli"]=1000)

# export model_path=/usr1/hahn2/mask/mask/outs/finetune/sst2/1e-5; export sparsity=0.0
# CUDA_VISIBLE_DEVICES=1 python run_glue.py \
#   --model_name_or_path $model_path \
#   --dataset_name $task \
#   --do_eval \
#   --output_dir null \
#   --max_seq_length 128 \
#   --initial_sparsity $sparsity \

export model_path=/usr1/hahn2/mask/mask/outs/mask_cloze/sst2/5e-4/0.05; export sparsity=0.05
CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path $model_path \
  --dataset_name $task \
  --do_eval \
  --output_dir null \
  --max_seq_length 128 \
  --initial_sparsity $sparsity \
  --cloze_task
