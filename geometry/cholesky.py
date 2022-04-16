# import tensorflow as tf
import torch #as tf
import numpy as np
# from utils.einsum import einsum
from torch import einsum



class _cholesky_solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        chol = torch.cholesky(H)
        xx = torch.cholesky_solve(b, chol)
        ctx.save_for_backward(chol, xx)

        return xx

    # see OptNet: https://arxiv.org/pdf/1703.00443.pdf
    @staticmethod
    def backward(ctx, dx):
        chol, xx = ctx.saved_tensors

        dz = torch.cholesky_solve(dx, chol)
        xs = torch.squeeze(xx,  -1)
        zs = torch.squeeze(dz, -1)
        dH = -einsum('...i,...j->...ij', xs, zs)

        return dH, dz
def cholesky_solve(H, b):
    return _cholesky_solve.apply(H,b)

def solve(H, b, max_update=1.0):
    """ Solves the linear system Hx = b, H > 0"""

    # small system, solve on cpu
    H = H.to(dtype=torch.float64) 
    b = b.to(dtype=torch.float64) 

    b = torch.unsqueeze(b, -1)
    x = cholesky_solve(H, b)

    # replaces nans and clip large updates
    bad_values = torch.isnan(x) 
    x = torch.where(bad_values, torch.zeros_like(x), x)
    x = torch.clamp(x, -max_update, max_update)

    x = torch.squeeze(x, -1)
    x = x.to(dtype=torch.float32) 
        
    return x



def __test__():
    import numpy as np 
    np.random.seed(0)
    M=np.random.uniform(size=(3,3))
    H=torch.tensor(M@M.transpose(-1,-2), requires_grad=True )

    b=torch.tensor(np.random.uniform(size=(3,) ), requires_grad=True )

    x= solve(H,b )

    x.backward(torch.ones_like(x) )


    print(f"H={H}, b={b}, x={x}, grad={H.grad, b.grad}")

if __name__=="__main__":
    __test__()
