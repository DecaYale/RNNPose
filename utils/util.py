import torch
import numpy as np
import collections



def freeze_params(params: dict, include: str = None, exclude: str = None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue
        remain_params.append(p)
    return remain_params


def freeze_params_v2(params: dict, include: str = None, exclude: str = None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False


def filter_param_dict(state_dict: dict, include: str = None, exclude: str = None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue
        res_dict[k] = p
    return res_dict

def modify_parameter_name_with_map(state_dict, parameteter_name_map=None):
    if parameteter_name_map is None:
        return state_dict
    for old,new in parameteter_name_map:
        for key in list(state_dict.keys()) :
            if old in key:
                new_key=key.replace(old, new)
                state_dict[new_key] = state_dict.pop(key)
    return state_dict

def load_pretrained_model_map_func(state_dict,parameteter_name_map = None, include:str=None, exclude:str=None):
    state_dict = filter_param_dict(state_dict, include, exclude)
    state_dict = modify_parameter_name_with_map(state_dict, parameteter_name_map)
    


def list_recursive_op(input_list, op):
    assert isinstance(input_list, list)

    for i, v in enumerate(input_list):
        if isinstance(v, list):
            input_list[i] = list_recursive_op(v, op)
        elif isinstance(v, dict):
            input_list[i] = dict_recursive_op(v, op)
        else:
            input_list[i] = op(v)

    return input_list


def dict_recursive_op(input_dict, op):
    assert isinstance(input_dict, dict)

    for k, v in input_dict.items():
        if isinstance(v, dict):
            input_dict[k] = dict_recursive_op(v, op)
        elif isinstance(v, (list,tuple) ):
            input_dict[k] = list_recursive_op(v, op)
        else:
            input_dict[k] = op(v)

    return input_dict

