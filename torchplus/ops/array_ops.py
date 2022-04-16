import ctypes
import math
import time
import torch
from typing import Optional


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


def roll(x: torch.Tensor, shift: int, dim: int = -1, fill_pad: Optional[int] = None):
    """
        shift<0: left roll
        shift>0: right roll
    """  
    device = x.device
    
    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift, device=device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim), device=device)), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim), device=device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift, device=device))], dim=dim) 