from functools import reduce

import torch

from layers import MaskedLinear, MaskedEmbedding


def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


def is_leaf_module(module):
    return len(list(module.children())) == 0


def calculate_sparsity(model):
    if isinstance(model, dict):
        zero_params = 0
        total_params = 0
        for k, v in model.items():
            zero_params += torch.sum(v == 0).item()
            total_params += v.numel()
        return zero_params / total_params
    else:
        zero_params = 0
        total_params = 0
        for n, m in model.named_modules():
            if is_leaf_module(m):
                if hasattr(m, "weight"):
                    total_params += m.weight.numel()
                    if isinstance(m, MaskedLinear):
                        zero_params += m.num_zeros
                    if isinstance(m, MaskedEmbedding):
                        zero_params += m.num_zeros
        return zero_params / total_params


def chain(*funcs):
    def chained_call(arg):
        return reduce(lambda r, f: f(r), funcs, arg)

    return chained_call


def get_mask(model):
    mask_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, MaskedLinear):
            mask_dict[n] = m.mask
    return mask_dict


def calculate_hamming_dist(prev_mask_dict, curr_mask_dict):
    assert set(prev_mask_dict.keys()) == set(curr_mask_dict.keys())
    keys = prev_mask_dict.keys()
    num_changed = sum([torch.sum(prev_mask_dict[k] != curr_mask_dict[k]).item() for k in keys])
    num_total = sum([v.numel() for v in prev_mask_dict.values()])
    return num_changed / num_total

