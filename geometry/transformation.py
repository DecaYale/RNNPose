import torch  
import numpy as np

# from core.config import cfg
from config.default import get_cfg
from .se3 import *
from .intrinsics import *
from . import projective_ops as pops
from . import cholesky

from .einsum import einsum

cholesky_solve = cholesky.solve


MIN_DEPTH = 0.1
MAX_RESIDUAL = 250.0

# can use both matrix or quaternions to represent rotations
DEFAULT_INTERNAL = 'matrix'


def clip_dangerous_gradients(x):
    return x


def jac_local_perturb(pt, fill=False):

    X, Y, Z = torch.split(pt,[1,1,1], dim=-1)  # torch.split(pt, [1, 1, 1], axis=-1)
    o, i = torch.zeros_like(X), torch.ones_like(X)
    if fill:
        j1 = torch.cat([i,  o,  o, o], dim=-1)
        j2 = torch.cat([o,  i,  o, o], dim=-1)
        j3 = torch.cat([o,  o,  i, o], dim=-1)
        j4 = torch.cat([o, -Z,  Y, o], dim=-1)
        j5 = torch.cat([Z,  o, -X, o], dim=-1)
        j6 = torch.cat([-Y,  X,  o, o],dim=-1)
    else:
        j1 = torch.cat([i,  o,  o], dim=-1)
        j2 = torch.cat([o,  i,  o], dim=-1)
        j3 = torch.cat([o,  o,  i], dim=-1)
        j4 = torch.cat([o, -Z,  Y], dim=-1)
        j5 = torch.cat([Z,  o, -X], dim=-1)
        j6 = torch.cat([-Y,  X,  o],dim=-1)
    jac = torch.stack([j1, j2, j3, j4, j5, j6], dim=-1)
    return jac


def cond_transform(cond, T1, T2):
    """ Return T1 if cond, else T2 """

    if T1.internal == 'matrix':
        mat = torch.cond(cond, lambda: T1.matrix(), lambda: T2.matrix())
        T = T1.__class__(matrix=mat, internal=T1.internal)

    elif T1.internal == 'quaternion':
        so3 = torch.cond(cond, lambda: T1.so3, lambda: T2.so3)
        translation = torch.cond(cond, lambda: T1.translation,
                              lambda: T2.translation)
        T = T1.__class__(so3=so3, translation=translation,
                         internal=T1.internal)
    return T


class SE3:
    def __init__(self, upsilon=None, matrix=None, so3=None, translation=None, eq=None, internal=DEFAULT_INTERNAL):
        self.eq = eq
        self.internal = internal

        if internal == 'matrix':
            if upsilon is not None:
                self.G = se3_matrix_expm(upsilon)
            elif matrix is not None:
                self.G = matrix
        else:
            raise NotImplementedError 

    def __call__(self, pt, jacobian=False):
        """ Transform set of points """

        if self.internal == 'matrix':

            pt = torch.cat([pt, torch.ones_like(pt[..., :1])],
                        dim=-1)  # convert to homogenous
            pt = einsum(self.eq, self.G[..., :3, :], pt)
        else:
            raise NotImplementedError

        if jacobian:
            jacobian = jac_local_perturb(pt)
            return pt, jacobian

        return pt

    def __mul__(self, other):
        if self.internal == 'matrix':
            G = torch.matmul(self.G, other.G)
            return self.__class__(matrix=G, internal=self.internal)
        else:
            raise NotImplementedError

    def identity_(self):
        if self.internal == 'matrix':
            shape=self.G.shape
            self.G=torch.eye(4, device=self.G.device).repeat([*shape[:-2],1,1])
        else:
            raise NotImplementedError


    def increment(self, upsilon):
        if self.internal == 'matrix':
            G = se3_matrix_increment(self.G, upsilon)
            return self.__class__(matrix=G, internal=self.internal)
        else:
            raise NotImplementedError

    def concat(self, other, axis=0):
        if self.internal == 'matrix':
            G = torch.concat([self.G, other.G], axis=axis)
        else:
            raise NotImplementedError


    def copy(self, stop_gradients=False):

        if self.internal == 'matrix':
            if stop_gradients:
                # return self.__class__(matrix=torch.stop_gradient(self.G), internal=self.internal)
                return self.__class__(matrix=self.G.detach(), internal=self.internal)
            else:
                return self.__class__(matrix=self.G, internal=self.internal)

        else:
            raise NotImplementedError

    def to_vec(self):
        return torch.concat([self.so3, self.translation], axis=-1)

    def inv(self):
        if self.internal == 'matrix':
            Ginv = se3_matrix_inverse(self.matrix())
            return self.__class__(matrix=Ginv, internal=self.internal)
        else:
            raise NotImplementedError

    def adj(self):
        if self.internal == 'matrix':
            R = self.G[..., :3, :3]
            t = self.G[..., :3, 3]
            A11 = R
            A12 = torch.matmul(hat(t), R)
            A21 = torch.zeros_like(A11)
            A22 = R
        else:
            raise NotImplementedError


        Ax = torch.concat([
            torch.concat([A11, A12], axis=-1),
            torch.concat([A21, A22], axis=-1)
        ], axis=-2)

        return Ax

    def logm(self):
        return se3_logm(self.so3, self.translation)

    def shape(self):
        # return torch.shape(self.so3)[:-1]
        if self.internal == 'matrix':
            my_shape = self.G.shape  # torch.shape(self.G)
        else:
            raise NotImplementedError

        return (my_shape[0], my_shape[1])

    def matrix(self, fill=True):
        if self.internal == 'matrix':
            return self.G
        else:
            raise NotImplementedError
       

    def transform(self, depth, intrinsics, valid_mask=False, return3d=False):
        
        # pt = pops.backproject(depth, intrinsics)
        pt = pops.backproject(depth, intrinsics)
        pt_new = self.__call__(pt)
        coords = pops.project(pt_new, intrinsics)
        if return3d:
            return coords, pt_new
        if valid_mask:
            vmask = (pt[..., -1] > MIN_DEPTH) & (pt_new[..., -1] > MIN_DEPTH)
            # vmask = torch.cast(vmask, torch.float32)[..., torch.newaxis]
            # vmask = vmask.to(dtype=torch.float32)[..., None, :,:] #BxKx1xHxW
            vmask = vmask.to(dtype=torch.float32)[..., :, :, None]  # BxKx1xHxW
            return coords, vmask
        return coords

    def induced_flow(self, depth, intrinsics, valid_mask=False):
        coords0 = pops.coords_grid(depth, homogeneous=False)

        if valid_mask:
            coords1, vmask = self.transform(
                depth, intrinsics, valid_mask=valid_mask)
            return coords1 - coords0, vmask
        coords1 = self.transform(depth, intrinsics, valid_mask=valid_mask)
        return coords1 - coords0

    def depth_change(self, depth, intrinsics):
        pt = pops.backproject(depth, intrinsics)
        pt_new = self.__call__(pt)
        return pt_new[..., -1] - pt[..., -1]
    
    def identity(self):
        """ Push identity transformation to start of collection """
        batch, frames = self.shape()
        if self.internal == 'matrix':
            # I = torch.eye(4, batch_shape=[batch, 1])
            I = torch.eye(4, dtype=self.G.dtype, device=self.G.device).repeat(
                [batch, 1, 1, 1])
            # return self.__class__(matrix=I, internal=self.internal, eq=self.eq)
            return self.__class__(matrix=I, internal=self.internal, eq=self.eq)
        else:
            raise NotImplementedError




class SE3Sequence(SE3):
    """ Stores collection of SE3 objects """

    def __init__(self, upsilon=None, matrix=None, so3=None, translation=None, eq= "aijk,ai...k->ai...j",internal=DEFAULT_INTERNAL):
        super().__init__(
            upsilon, matrix, so3, translation, internal=internal, eq=eq)

        # self.eq = "aijk,ai...k->ai...j"
    def __call__(self, pt, inds=None, jacobian=False):
        if self.internal == 'matrix':
            return super().__call__(pt, jacobian=jacobian)
        else:
            raise NotImplementedError


    def gather(self, inds):
        if self.internal == 'matrix':
            G = torch.index_select(self.G, index=inds, dim=1)
            return SE3Sequence(matrix=G, internal=self.internal)
        else:
            raise NotImplementedError

    # def append_identity(self):
    #     """ Push identity transformation to start of collection """
    #     batch, frames = self.shape()
    #     if self.internal == 'matrix':
    #         # I = torch.eye(4, batch_shape=[batch, 1])
    #         I = torch.eye(4, dtype=self.G.dtype, device=self.G.device).repeat(
    #             [batch, 1, 1, 1])

    #         G = torch.cat([I, self.G], dim=1)
    #         return SE3Sequence(matrix=G, internal=self.internal)
    #     else:
    #         raise NotImplementedError

    def reprojction_optim(self,
                       target,
                       weight,
                       depth,
                       intrinsics,
                       num_iters=2,
                       depth_img_coords=None
                       ):

        target = clip_dangerous_gradients(target).to(dtype=torch.float64)
        weight = clip_dangerous_gradients(weight).to(dtype=torch.float64)

        X0 = pops.backproject(depth, intrinsics, depth_coords=depth_img_coords)
        w = weight[..., None] 

        lm_lmbda = get_cfg("LM").LM_LMBDA
        ep_lmbda = get_cfg("LM").EP_LMBDA

        T = self.copy(stop_gradients=False)
        for i in range(num_iters):
            ### compute the jacobians of the transformation ###
            X1, jtran = T(X0, jacobian=True)
            x1, (jproj, jkvec) = pops.project(X1, intrinsics, jacobian=True)

            v = (X0[..., -1] > MIN_DEPTH) & (X1[..., -1] > MIN_DEPTH)
            # v = v.to(dtype=torch.float32)[..., None, None]
            v = v.to(dtype=torch.float64)[..., None, None]

            ### weighted gauss-newton update ###
            J = einsum('...ij,...jk->...ik', jproj.to(dtype=torch.float64), jtran.to(dtype=torch.float64 ))  

            H = einsum('ai...j,ai...k->aijk', v*w*J, J)
            b = einsum('ai...j,ai...->aij', v*w*J, target-x1)

            ### add dampening and apply increment ###
            H += ep_lmbda*torch.eye(6, dtype=H.dtype, device=H.device) + lm_lmbda*H*torch.eye(6,dtype=H.dtype, device=H.device)
            try:
                delta_upsilon = cholesky_solve(H, b)
            except:
                # print(w.shape,v.shape, w.mean(), v.mean(),H,b, '!!!!')
                raise
            T = T.increment(delta_upsilon)

        # update
        if self.internal == 'matrix':
            self.G = T.matrix()
            T = SE3Sequence(
                matrix=T.matrix(), internal=self.internal)
        else:
            raise NotImplementedError

        return T


    def transform(self, depth, intrinsics, valid_mask=False, return3d=False):
        return super().transform(depth, intrinsics, valid_mask, return3d)
