from functools import reduce

import torch
import torch.nn as nn

from layers import MaskedLinear


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
    num_changed = sum([
        torch.sum(prev_v != curr_v).item() \
                for prev_v, curr_v in zip(prev_mask_dict.values(), curr_mask_dict.values())
                ])
    num_total = sum([v.numel() for v in prev_mask_dict.values()])
    return num_changed / num_total


def maskfy_model(model, initial_sparsity, init_scale=0.02, threshold=0.01):
    if initial_sparsity != 0.0:  # load from saved
        for n, p in model.named_parameters():
            p.requires_grad = False
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                masked_linear = MaskedLinear(m.weight,
                                             m.bias,
                                             mask_scale=init_scale,
                                             threshold=threshold,
                                             initial_sparsity=initial_sparsity,
                                             )
                masked_linear.mask_real.requires_grad = True
                masked_linear.bias.requires_grad = False
                recursive_setattr(model, n, masked_linear)
            # elif isinstance(m, nn.Embedding):
            #     masked_embedding = MaskedEmbedding(m.weight,
            #                                        m.padding_idx,
            #                                        mask_scale=args.init_scale,
            #                                        threshold=args.threshold,
            #                                        initial_sparsity=args.initial_sparsity
            #                                        )
            #     masked_embedding.mask_real.requires_grad = True
            #     recursive_setattr(model, n, masked_embedding)
        print(f" ========== Initial sparsity: {calculate_sparsity(model)} ==========")
    return model
