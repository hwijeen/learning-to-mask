#!/bin/bash
export task=$1
declare -A ft_lr=(["mrpc"]=3e-5 ["rte"]=3e-5 ["cola"]=1e-5 ["sst2"]=2e-5 ["qnli"]=2e-5 ["qqp"]=1e-5 ["stsb"]=1e-5 ["mnli"]=3e-5)
declare -A mask_lr=(["mrpc"]=5e-4 ["rte"]=5e-4 ["cola"]=5e-4 ["sst2"]=5e-4 ["qnli"]=5e-4 ["qqp"]=5e-4 ["stsb"]=5e-4 ["mnli"]=5e-4)


declare -A eval_tasks=(\
    ["sst2"]="imdb yelp_polarity amazon_polarity" \
    # ["mnli"]="SNLI SICK"\
    # ["qnli"]="NewsQA"\
    ["RTE"]="mrpc qqp hans"\
    ["mrpc"]="qqp rte hans"\
    ["qqp"]="mrpc rte hans"\
    # ["cola"]="grammar_test"\
)

# # finetune
for eval_t in ${eval_tasks[$task]}
do
    export model_path=/projects/tir6/strubell/hahn2/mask/outs/finetune/${task}/${ft_lr[$task]}
    echo $model_path $eval_t
    python run_glue.py \
      --model_name_or_path $model_path \
      --dataset_name $eval_t \
      --do_eval \
      --output_dir eval_result/$task/$eval_t \
      --max_seq_length 128 \
      --initial_sparsity 0.0
done

# # mask
# for eval_t in ${eval_tasks[$task]}
# do
#     export model_path=/projects/tir6/strubell/hahn2/mask/outs/mask_cloze/${task}/${mask_lr[$task]}
#     echo $model_path $eval_t
#     python run_glue.py \
#       --model_name_or_path $model_path \
#       --dataset_name $eval_t \
#       --do_eval \
#       --output_dir eval_result \
#       --max_seq_length 128 \
#       --initial_sparsity 0.0 \
# done
