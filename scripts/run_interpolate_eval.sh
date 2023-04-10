#!/bin/bash
export task1=$1
export task2=$2
export alpha=$3
declare -A ft_lr=(["mrpc"]=3e-5 ["rte"]=3e-5 ["cola"]=1e-5 ["sst2"]=2e-5 ["qnli"]=2e-5 ["qqp"]=1e-5 ["stsb"]=1e-5 ["mnli"]=3e-5)
declare -A mask_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)
declare -A fisher_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)


export model_path1=outs/mask_cloze/${task1}/${mask_lr[$task1]}/0.05
export model_path2=outs/mask_cloze/${task2}/${mask_lr[$task2]}/0.05
export interpolated_path=/projects/tir6/strubell/hahn2/mask/outs/mask_cloze/interpolated/${task1}_${task2}_${alpha}

python analysis.py \
  --model_name_or_path1 $model_path1 \
  --model_name_or_path2 $model_path2 \
  --alpha $alpha \
  --out_path $interpolated_path

for eval_t in $task1 $task2; do
    python run_glue.py \
        --model_name_or_path $interpolated_path \
        --dataset_name $eval_t \
        --do_eval \
        --per_device_eval_batch_size 128 \
        --output_dir eval_result/interpolated/${task1}_${task2}_${alpha}/${eval_t} \
        --max_seq_length 128 \
        --initial_sparsity 0.05 \
        --cloze_task
done
rm -rf $interpolated_path
