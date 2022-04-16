from torchplus.train import learning_schedules
from torchplus.train import optim
import torch
from torch import nn
from torchplus.train.fastai_optim import OptimWrapper, FastAIMixedOptim
from functools import partial


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))

# return a list of smallest modules dy


def flatten_model(m):
    if m is None:
        return []
    return sum(
        map(flatten_model, m.children()), []) if num_children(m) else [m]


# def get_layer_groups(m): return [nn.Sequential(*flatten_model(m))]
def get_layer_groups(m): return [nn.ModuleList(flatten_model(m))]

def get_voxeLO_net_layer_groups(net):
    vfe_grp = get_layer_groups(net)#[0]

    other_grp = get_layer_groups(nn.Sequential(net._rotation_loss,
                    net._translation_loss,
                    net._pyramid_rotation_loss,
                        net._pyramid_translation_loss,
                     net._consistency_loss, 
                     ))

    return [vfe_grp, mfe_grp, op_grp,other_grp]


def get_voxeLO_net_layer_groups(net):
    vfe_grp = get_layer_groups(net.voxel_feature_extractor)#[0]
    mfe_grp = get_layer_groups(net.middle_feature_extractor)#[0]
    op_grp = get_layer_groups(net.odom_predictor)#[0]

    # other_grp = get_layer_groups(net._rotation_loss) +  \
    #     get_layer_groups(net._translation_loss) \
    #         + get_layer_groups(net._pyramid_rotation_loss) \
    #             + get_layer_groups(net._pyramid_translation_loss) \
    #                 + get_layer_groups(net._consistency_loss)\
    other_grp = get_layer_groups(nn.Sequential(net._rotation_loss,
                    net._translation_loss,
                    net._pyramid_rotation_loss,
                        net._pyramid_translation_loss,
                     net._consistency_loss, 
                     ))

    return [vfe_grp, mfe_grp, op_grp,other_grp]


def build(optimizer_config, net, name=None, mixed=False, loss_scale=512.0):

    optimizer_type = list(optimizer_config.keys())[0]
    print("Optimizer:", optimizer_type)
    
    optimizer=None

    if optimizer_type == 'rms_prop_optimizer':
        config=optimizer_config.rms_prop_optimizer
        optimizer_func=partial(
            torch.optim.RMSprop,
            alpha=config.decay,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon)

    if optimizer_type == 'momentum_optimizer':
        config=optimizer_config.momentum_optimizer
        optimizer_func=partial(
            torch.optim.SGD,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon)

    if optimizer_type == 'adam_optimizer':
        config=optimizer_config.adam_optimizer
        if optimizer_config.fixed_weight_decay:
            optimizer_func=partial(
                torch.optim.Adam, betas=(0.9, 0.99), amsgrad=config.amsgrad)
        else:
            # regular adam
            optimizer_func=partial(
                torch.optim.Adam, amsgrad=config.amsgrad)

    optimizer=OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(net),
        # get_voxeLO_net_layer_groups(net),
        wd=config.weight_decay,
        true_wd=optimizer_config.fixed_weight_decay,
        bn_wd=True)
    print(hasattr(optimizer, "_amp_stash"), '_amp_stash')
    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        raise ValueError('torch don\'t support moving average')
    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name=optimizer_type
    else:
        optimizer.name=name
    return optimizer
