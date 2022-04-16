import torch
import numpy as np
# from utils.einsum import einsum
from .einsum import einsum

def intrinsics_vec_to_matrix(kvec):
    fx, fy, cx, cy = torch.unbind(kvec, dim=-1)
    z = torch.zeros_like(fx)
    o = torch.ones_like(fx)

    K = torch.stack([fx, z, cx, z, fy, cy, z, z, o], dim=-1)
    K = torch.reshape(K, list(kvec.shape)[:-1] + [3,3])
    return K

def intrinsics_matrix_to_vec(kmat):
    fx = kmat[..., 0, 0]
    fy = kmat[..., 1, 1]
    cx = kmat[..., 0, 2]
    cy = kmat[..., 1, 2]
    return torch.stack([fx, fy, cx, cy], dim=-1)

def update_intrinsics(intrinsics, delta_focal):
    kvec = intrinsics_matrix_to_vec(intrinsics)
    fx, fy, cx, cy = torch.unstack(kvec, num=4, axis=-1)
    df = torch.squeeze(delta_focal, -1)

    # update the focal lengths
    fx = torch.exp(df) * fx
    fy = torch.exp(df) * fy

    kvec = torch.stack([fx, fy, cx, cy], axis=-1)
    kmat = intrinsics_vec_to_matrix(kvec)
    return kmat

def rescale_depth(depth, downscale=4):
    depth = depth[:,None]
    new_shape = [depth.shape[-2]//downscale, depth.shape[-1]//downscale]
    depth = torch.nn.functional.interpolate(depth, new_shape, mode='nearest')
    return torch.squeeze(depth, dim=1)

def rescale_depth_and_intrinsics(depth, intrinsics, downscale=4):
    sc = torch.tensor([1.0/downscale, 1.0/downscale, 1.0], dtype=torch.float32, device=depth.device)
    intrinsics = einsum('...ij,i->...ij', intrinsics, sc)
    depth = rescale_depth(depth, downscale=downscale)
    return depth, intrinsics

def rescale_depths_and_intrinsics(depth, intrinsics, downscale=4):
    batch, frames, height, width = [depth.shape[i] for i in range(4)]
    depth = torch.reshape(depth, [batch*frames, height, width])
    depth, intrinsics = rescale_depth_and_intrinsics(depth, intrinsics, downscale)
    depth = torch.reshape(depth,
        [batch, frames]+list(depth.shape)[1:])
    return depth, intrinsics
