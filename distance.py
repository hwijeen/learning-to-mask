import argparse
import os
from copy import deepcopy
from collections import defaultdict
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from layers import MaskedLinear, MaskedEmbedding
from utils import recursive_setattr, calculate_sparsity, is_leaf_module, get_mask, calculate_hamming_dist


logging.getLogger("transformers").setLevel(logging.ERROR)


# parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="outs/mask_cloze/")
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
        elif isinstance(m, nn.Embedding):
            masked_embedding = MaskedEmbedding(m.weight,
                                               m.padding_idx,
                                               mask_scale=0.02,
                                               threshold=0.01,
                                               initial_sparsity=0.05,
                                               )
            masked_embedding.mask_real.requires_grad = True
            recursive_setattr(model, n, masked_embedding)
    print("Loading from saved model: ", model_name_or_path)
    state_dict = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    print(f"========== Initial sparsity: {calculate_sparsity(model)} ==========\n\n")
    return model


def get_penultimate_leaf_folders(path):
    leaf_folders = []
    for root, dirs, files in os.walk(path):
        if not dirs:
            leaf_folders.append(os.path.dirname(root))
    return leaf_folders

def filter_paths(paths, keyword:list, not_contain:list=None):
    if not isinstance(keyword, list):
        keyword = [keyword]
    paths = [path for path in paths if all([k in path for k in keyword])]
    if not_contain is not None:
        paths = [path for path in paths if all([not_contain not in path for not_contain in not_contain])]
    return paths

def gather_distance(models, model_paths):
    results = defaultdict(dict)
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]

            mask_dict1 = get_mask(model1)
            mask_dict2 = get_mask(model2)

            dist = calculate_hamming_dist(mask_dict1, mask_dict2)
            model_path1 = model_paths[i]
            model_path2 = model_paths[j]
            results[model_path1][model_path2] = dist

    # make results into a symettric matrix
    for model_path1 in model_paths:
        for model_path2 in model_paths:
            if model_path1 == model_path2:
                results[model_path1][model_path2] = 0.0
            elif model_path2 not in results[model_path1]:
                results[model_path1][model_path2] = results[model_path2][model_path1]
    return results


def save_heatmap(t, model_paths):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(t, annot=True, xticklabels=model_paths, yticklabels=model_paths, vmin=0, vmax=1)
    plt.title(model_paths)
    plt.savefig("distance.png")

model_paths = get_penultimate_leaf_folders(args.model_path)
model_paths = filter_paths(model_paths, ["0.9", "42", "5e-5"], ["qqp"])
# model_paths = filter_paths(model_paths, ["0.05", "42", "5e-4"], ["qqp"])
# model_paths = filter_paths(model_paths, ["0.9", "5e-5", "sst2"])
models = [load_model(model_path) for model_path in model_paths]

results = gather_distance(models, model_paths)

t = torch.tensor([[results[model_path1][model_path2] for model_path2 in model_paths] for model_path1 in model_paths])
print(t)
save_heatmap(t, model_paths)
