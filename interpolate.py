import argparse
import os
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from layers import MaskedLinear
from utils import recursive_setattr, calculate_sparsity, is_leaf_module


# parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path1", type=str, default="outs/mask_cloze/qqp/5e-4/0.05")
parser.add_argument("--model_name_or_path2", type=str, default="outs/mask_cloze/sst2/5e-4/0.05")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--out_path")
args = parser.parse_args()

def load_model(model_name_or_path):
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    for n, p in model.named_parameters():
        p.requires_grad = False
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            masked_linear = MaskedLinear(m.weight,
                                         m.bias,
                                         mask_scale=0.02,
                                         threshold=0.01,
                                         initial_sparsity=0.05,
                                         )
            masked_linear.mask_real.requires_grad = True
            masked_linear.bias.requires_grad = False
            recursive_setattr(model, n, masked_linear)
    print(f"\n\n ========== Initial sparsity: {calculate_sparsity(model)} ==========\n\n")
    print("Loading from saved model: ", model_name_or_path)
    state_dict = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    print(f"\n\n ========== Initial sparsity: {calculate_sparsity(model)} ==========\n\n")
    return model


def apply_mask(model):
    for n, m in model.named_modules():
        if isinstance(m, MaskedLinear):
            m.weight *= m.mask
    return model

# def interpolate(model1, model2, alpha):
#     for (n1, m1), (n2, m2) in zip(model1.named_modules(), model2.named_modules()):
#         if not is_leaf_module(m1) or not is_leaf_module(m2):
#             continue
#         assert n1 == n2
#         if isinstance(m1, MaskedLinear) and isinstance(m2, MaskedLinear):
#             m1.weight = nn.Parameter(m1.weight * alpha + m2.weight * (1 - alpha))
#         if isinstance(m1, nn.Embedding) and isinstance(m2, nn.Embedding):
#             m1.weight = nn.Parameter(m1.weight * alpha + m2.weight * (1 - alpha))
#     return model1

def interpolate(model1, model2, alpha):
    for (n1, m1), (n2, m2) in zip(model1.named_modules(), model2.named_modules()):
        if not is_leaf_module(m1) or not is_leaf_module(m2):
            continue
        assert n1 == n2
        if isinstance(m1, MaskedLinear) and isinstance(m2, MaskedLinear):
            m1.mask_real = nn.Parameter(m1.mask_real * alpha + m2.mask_real * (1 - alpha))
        # if isinstance(m1, nn.Embedding) and isinstance(m2, nn.Embedding):
        #     m1.weight = nn.Parameter(m1.mask_real * alpha + m2.mask_real * (1 - alpha))
    return model1


def copy_tokenizer(args):
    os.system(f"cp {args.model_name_or_path1}/added_tokens.json {args.out_path}")
    os.system(f"cp {args.model_name_or_path1}/special_tokens_map.json {args.out_path}")
    os.system(f"cp {args.model_name_or_path1}/tokenizer.json {args.out_path}")
    os.system(f"cp {args.model_name_or_path1}/tokenizer_config.json {args.out_path}")
    os.system(f"cp {args.model_name_or_path1}/vocab.txt {args.out_path}")


#"./outs/mask_cloze/sst2/5e-4/0.05"
_, type1, task1, lr1, sparsity1 = args.model_name_or_path1.split("/")
# model1 = apply_mask(load_model(args.model_name_or_path1))
model1 = load_model(args.model_name_or_path1)

_, type2, task2, lr2, sparsity2 = args.model_name_or_path2.split("/")
# model2 = apply_mask(load_model(args.model_name_or_path2))
model2 = load_model(args.model_name_or_path2)

interpolated_model = interpolate(model1, model2, args.alpha)
# for (n1, p1), (n2, p2) in zip(model2.named_parameters(), interpolated_model.named_parameters()):
#     assert n1 == n2
#     assert torch.all(torch.eq(p1, p2))
interpolated_model.save_pretrained(args.out_path)
copy_tokenizer(args)
print(f"Model saved to {args.out_path}")

