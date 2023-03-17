#!/bin/bash
export model_type=$1
export task=$2
declare -A ft_lr=(["mrpc"]=3e-5 ["rte"]=3e-5 ["cola"]=1e-5 ["sst2"]=2e-5 ["qnli"]=2e-5 ["qqp"]=1e-5 ["stsb"]=1e-5 ["mnli"]=3e-5)
declare -A mask_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)
declare -A fisher_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)


declare -A eval_tasks=(\
    ["sst2"]="imdb yelp_polarity amazon_polarity" \
    # ["mnli"]="mnli snli sick"\
    ["mnli"]="mnli sick"\
    ["qnli"]="newsqa"\
    ["rte"]="mrpc qqp hans"\
    ["mrpc"]="qqp rte hans"\
    ["qqp"]="mrpc rte hans"\
    ["cola"]="grammar_test"\
)


# bash statement if model_type is "finetune"
for eval_t in ${eval_tasks[$task]}
do
    if [ $model_type = "finetune" ]; then
        export model_path=/projects/tir6/strubell/hahn2/mask/outs/finetune/${task}/${ft_lr[$task]}
		python run_glue.py \
		  --model_name_or_path $model_path \
		  --dataset_name $eval_t \
		  --do_eval \
		  --output_dir eval_result/$model_type/$task/$eval_t \
		  --max_seq_length 128 \
		  --initial_sparsity 0.0
    elif [ $model_type = "mask_cloze" ]; then
        export model_path=/projects/tir6/strubell/hahn2/mask/outs/mask_cloze/${task}/${mask_lr[$task]}/0.05
		python run_glue.py \
		  --model_name_or_path $model_path \
		  --dataset_name $eval_t \
		  --do_eval \
		  --output_dir eval_result/$model_type/$task/$eval_t \
		  --max_seq_length 128 \
		  --initial_sparsity 0.05 \
		  --cloze_task
    elif [ $model_type = "fisher" ]; then
        export model_path=/projects/tir6/strubell/hahn2/mask/outs/fisher/${task}/${fisher_lr[$task]}/0.05
		python run_glue.py \
		  --model_name_or_path $model_path \
		  --dataset_name $eval_t \
		  --do_eval \
		  --output_dir eval_result/$model_type/$task/$eval_t \
		  --max_seq_length 128 \
		  --initial_sparsity 0.05 \
		  --cloze_task
    fi
done
