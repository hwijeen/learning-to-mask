#!/bin/bash
export model_type=$1
export src_task=$2
export tgt_task=$3
declare -A ft_lr=(["mrpc"]=3e-5 ["rte"]=3e-5 ["cola"]=1e-5 ["sst2"]=2e-5 ["qnli"]=2e-5 ["qqp"]=1e-5 ["stsb"]=1e-5 ["mnli"]=3e-5)
declare -A mask_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)
declare -A fisher_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)


if [ $model_type = "finetune" ]; then
    export model_path=/projects/tir6/strubell/hahn2/mask/outs/finetune/${src_task}/${ft_lr[$src_task]}
    python run_glue.py \
      --model_name_or_path $model_path \
      --dataset_name $tgt_task \
      --do_eval \
      --output_dir eval_result/$model_type/$src_task/$tgt_task \
      --max_seq_length 128 \
      --initial_sparsity 0.0
elif [ $model_type = "mask_cloze" ]; then
    export model_path=/projects/tir6/strubell/hahn2/mask/outs/mask_cloze/${src_task}/${mask_lr[$src_task]}/0.05
    python run_glue.py \
      --model_name_or_path $model_path \
      --dataset_name $tgt_task \
      --do_eval \
      --output_dir eval_result/$model_type/$src_task/$tgt_task \
      --max_seq_length 128 \
      --initial_sparsity 0.05 \
      --cloze_task
elif [ $model_type = "fisher" ]; then
    export model_path=/projects/tir6/strubell/hahn2/mask/outs/fisher/${src_task}/${fisher_lr[$src_task]}/0.05
    python run_glue.py \
      --model_name_or_path $model_path \
      --dataset_name $tgt_task \
      --do_eval \
      --output_dir eval_result/$model_type/$src_task/$tgt_task \
      --max_seq_length 128 \
      --initial_sparsity 0.05 \
      --cloze_task
fi
