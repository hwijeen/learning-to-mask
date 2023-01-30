from functools import reduce

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
