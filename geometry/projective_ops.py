import numpy as np
import torch 

# from utils.einsum import einsum
from torch import einsum


# MIN_DEPTH = 0.1
MIN_DEPTH = 0.01

def normalize_coords_grid(coords):
    """ normalize the coordinates to [-1,1]

    Args:
        coords: BxKxHxWx2
    """
    coords=coords.clone()
    B,K,H,W,_ = coords.shape

    coords[...,0] = 2*coords[...,0]/(W-1)-1
    coords[...,1] = 2*coords[...,1]/(H-1)-1

    return coords

def coords_grid(ref, homogeneous=True):
    """ grid of pixel coordinates """
    shape = ref.shape

    yy, xx = torch.meshgrid(torch.arange(shape[-2], device=ref.device), torch.arange(shape[-1], device=ref.device) )

    xx = xx.to(dtype=torch.float32)
    yy = yy.to(dtype=torch.float32)

    if homogeneous:
        coords = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    else:
        coords = torch.stack([xx, yy], dim=-1)

    new_shape = [1]*len(shape[:-2]) +  list(shape[-2:]) + [-1]
    coords = torch.reshape(coords, new_shape)

    tile = list(shape[:-2])+ [1,1,1]
    coords = coords.repeat(tile)
    return coords # BxKxHxWx2


def extract_and_reshape_intrinsics(intrinsics, shape=None):
    """ Extracts (fx, fy, cx, cy) from intrinsics matrix """

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    if shape is not None:
        batch = list(fx.shape[:1])
        fillr = [1]*len(shape[1:]) 
        k_shape = batch+fillr

        fx = torch.reshape(fx, k_shape)
        fy = torch.reshape(fy, k_shape)
        cx = torch.reshape(cx, k_shape)
        cy = torch.reshape(cy, k_shape)

    return (fx, fy, cx, cy)


def backproject(depth, intrinsics, jacobian=False, depth_coords=None):
    """ backproject depth map to point cloud """
    #depth_coords: (BxKxHxWx2)

    if depth_coords is None:
        coords = coords_grid(depth, homogeneous=True)
        x, y, _ = torch.unbind(coords, axis=-1)
    else:
        x, y =  torch.unbind(depth_coords, axis=-1)

    x_shape = x.shape 
    
    fx, fy, cx, cy = extract_and_reshape_intrinsics(intrinsics, x_shape) #Bx1x1x1

    Z = depth  #BxKxHxW
    X = Z * (x - cx) / fx
    Y = Z * (y - cy) / fy 
    points = torch.stack([X, Y, Z], axis=-1)

    if jacobian:
        o = torch.zeros_like(Z) # used to fill in zeros

        # jacobian w.r.t (fx, fy) , of shape BxKxHxWx4x1
        jacobian_intrinsics = torch.stack([
            torch.stack([-X / fx], dim=-1),
            torch.stack([-Y / fy], dim=-1),
            torch.stack([o], dim=-1),
            torch.stack([o], dim=-1)], axis=-2)

        return points, jacobian_intrinsics
    
    return points
    # return points, coords


def project(points, intrinsics, jacobian=False):
    
    """ project point cloud onto image """
    X, Y, Z = torch.unbind(points, axis=-1)
    Z = torch.clamp(Z, min=MIN_DEPTH)

    x_shape = X.shape
    fx, fy, cx, cy = extract_and_reshape_intrinsics(intrinsics, x_shape)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    coords = torch.stack([x, y], axis=-1)

    if jacobian:
        o = torch.zeros_like(x) # used to fill in zeros
        zinv1 = torch.where(Z <= MIN_DEPTH+.01, torch.zeros_like(Z), 1.0 / Z)
        zinv2 = torch.where(Z <= MIN_DEPTH+.01, torch.zeros_like(Z), 1.0 / Z**2)

        # jacobian w.r.t (X, Y, Z)
        jacobian_points = torch.stack([
            torch.stack([fx * zinv1, o, -fx * X * zinv2], axis=-1),
            torch.stack([o, fy * zinv1, -fy * Y * zinv2], axis=-1)], axis=-2)

        # jacobian w.r.t (fx, fy)
        jacobian_intrinsics = torch.stack([
            torch.stack([X * zinv1], axis=-1),
            torch.stack([Y * zinv1], axis=-1),], axis=-2)

        return coords, (jacobian_points, jacobian_intrinsics)

    return coords
