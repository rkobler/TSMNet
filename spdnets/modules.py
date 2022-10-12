import math
from typing import Tuple
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.types import Number
import torch.nn as nn
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel, Sphere
from . import functionals

class Conv2dWithNormConstraint(torch.nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithNormConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithNormConstraint, self).forward(x)


class LinearWithNormConstraint(torch.nn.Linear):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LinearWithNormConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithNormConstraint, self).forward(x)


class MySquare(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.square()

class MyLog(torch.nn.Module):
    def __init__(self, eps = 1e-4):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return torch.log(x + self.eps)


class MyConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MyConv2d, self).__init__(*args, **kwargs)
        
        self.convshape = self.weight.shape
        w0 = self.weight.data.flatten(start_dim=1)
        self.weight = ManifoldParameter(w0 / w0.norm(dim=-1, keepdim=True), manifold=Sphere())

    def forward(self, x):
        return self._conv_forward(x, self.weight.view(self.convshape), self.bias)


class UnitNormLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super(UnitNormLinear, self).__init__(*args, **kwargs)
        
        w0 = self.weight.data.flatten(start_dim=1)
        self.weight = ManifoldParameter(w0 / w0.norm(dim=-1, keepdim=True), manifold=Sphere())

    def forward(self, x):
        return super().forward(x)


class MyLinear(nn.Module):
    def __init__(self, shape : Tuple[int, ...] or torch.Size, bias: bool = True, **kwargs):
        super().__init__()

        self.W = Parameter(torch.empty(shape, **kwargs))

        if bias:
            self.bias = Parameter(torch.empty((*shape[:-2], shape[-1]), **kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def forward(self, X : Tensor) -> Tensor:
        A = (X.unsqueeze(-2) @ self.W).squeeze(-2)
        if self.bias is not None:
            A += self.bias
        return A

    @torch.no_grad()
    def reset_parameters(self):
        # kaiming initialization std2uniformbound * gain * fan_in
        bound = math.sqrt(3) * 1. / math.sqrt(self.W.shape[-2])
        self.W.data.uniform_(-bound,bound)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.W.shape[-2])
            self.bias.data.uniform_(-bound, bound)


class Encode2DPosition(nn.Module):
    """
    Encodes the 2D position of a 2D CNN or 2D image 
    as additional channels.
    Input: (batch, chans, height, width)
    Output: (batch, chans+2, height, width)
    """
    def __init__(self, flatten = True):
        super().__init__()
        self.flatten = flatten
    
    def forward(self, X : Tensor) -> Tensor:
        pos1 = torch.arange(X.shape[-2])[None,None,:,None].tile((X.shape[0],1, 1, X.shape[-1])) / X.shape[-2]
        pos2 = torch.arange(X.shape[-1])[None,None,None,:].tile((X.shape[0],1, X.shape[-2], 1)) / X.shape[-1]

        Z = torch.cat((X, pos1, pos2),dim=1)
        if self.flatten:
            Z = Z.flatten(start_dim=-2)

        return Z


class CovariancePool(nn.Module):
    def __init__(self, alpha = None, unitvar = False):
        super().__init__()
        self.pooldim = -1
        self.chandim = -2
        self.alpha = alpha
        self.unitvar = unitvar
    
    def forward(self, X : Tensor) -> Tensor:
        X0 = X - X.mean(dim=self.pooldim, keepdim=True)
        if self.unitvar:
            X0 = X0 / X0.std(dim=self.pooldim, keepdim=True)
            X0.nan_to_num_(0)

        C = (X0 @ X0.transpose(-2, -1)) / X0.shape[self.pooldim]
        if self.alpha is not None:
            Cd = C.diagonal(dim1=self.pooldim, dim2=self.pooldim-1)
            Cd += self.alpha
        return C


class ReverseGradient(nn.Module):
    def __init__(self, scaling = 1.):
        super().__init__()
        self.scaling_ = scaling
    
    def forward(self, X : Tensor) -> Tensor:
        return functionals.reverse_gradient.apply(X, self.scaling_)


class BiMap(nn.Module):
    def __init__(self, shape : Tuple[int, ...] or torch.Size, W0 : Tensor = None, manifold='stiefel', **kwargs):
        super().__init__()

        if manifold == 'stiefel':
            assert(shape[-2] >= shape[-1])
            mf = Stiefel()
        elif manifold == 'sphere':
            mf = Sphere()
            shape = list(shape)
            shape[-1], shape[-2] = shape[-2], shape[-1]
        else:
            raise NotImplementedError()

        # add constraint (also initializes the parameter to fulfill the constraint)
        self.W = ManifoldParameter(torch.empty(shape, **kwargs), manifold=mf)

        # optionally initialize the weights (initialization has to fulfill the constraint!)
        if W0 is not None:
            self.W.data = W0 # e.g., self.W = torch.nn.init.orthogonal_(self.W)
        else:
            self.reset_parameters()
    
    def forward(self, X : Tensor) -> Tensor:
        if isinstance(self.W.manifold, Sphere):
            return self.W @ X @ self.W.transpose(-2,-1)
        else:
            return self.W.transpose(-2,-1) @ X @ self.W

    @torch.no_grad()
    def reset_parameters(self):
        if isinstance(self.W.manifold, Stiefel):
            # uniform initialization on stiefel manifold after theorem 2.2.1 in Chikuse (2003): statistics on special manifolds
            W = torch.rand(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            self.W.data = W @ functionals.sym_invsqrtm.apply(W.transpose(-1,-2) @ W)
        elif isinstance(self.W.manifold, Sphere):
            W = torch.empty(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            # kaiming initialization std2uniformbound * gain * fan_in
            bound = math.sqrt(3) * 1. / W.shape[-1]
            W.uniform_(-bound, bound)
            # constraint has to be satisfied
            self.W.data = W / W.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError()


class ReEig(nn.Module):
    def __init__(self, threshold : Number = 1e-4):
        super().__init__()
        self.threshold = Tensor([threshold])

    def forward(self, X : Tensor) -> Tensor:
        return functionals.sym_reeig.apply(X, self.threshold)


class LogEig(nn.Module):
    def __init__(self, ndim, tril=True):
        super().__init__()

        self.tril = tril
        if self.tril:
            ixs_lower = torch.tril_indices(ndim,ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.ixs = torch.cat((ixs_diag[None,:].tile((2,1)), ixs_lower), dim=1)
        self.ndim = ndim

    def forward(self, X : Tensor) -> Tensor:
        return self.embed(functionals.sym_logm.apply(X))

    def embed(self, X : Tensor) -> Tensor:
        if self.tril:
            x_vec = X[...,self.ixs[0],self.ixs[1]]
            x_vec[...,self.ndim:] *= math.sqrt(2)
        else:
            x_vec = X.flatten(start_dim=-2)
        return x_vec


class TrilEmbedder(nn.Module):

    def forward(self, X : Tensor) -> Tensor:
        
        ndim = X.shape[-1]
        ixs_lower = torch.tril_indices(ndim,ndim, offset=-1)
        ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
        ixs = torch.cat((ixs_diag[None,:].tile((2,1)), ixs_lower), dim=1)

        x_vec = X[...,ixs[0],ixs[1]]
        x_vec[...,ndim:] *= math.sqrt(2)
        return x_vec

    def inverse_transform(self, x_vec: Tensor) -> Tensor:

        ndim = int(-.5 + math.sqrt(.25 + 2*x_vec.shape[-1])) # c*(c+1)/2 = nts
        ixs_lower = torch.tril_indices(ndim,ndim, offset=-1)
        ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)

        X = torch.zeros(x_vec.shape[:-1] + (ndim, ndim), device=x_vec.device, dtype=x_vec.dtype)

        # off diagonal elements
        X[...,ixs_lower[0],ixs_lower[1]] = x_vec[...,ndim:] / math.sqrt(2)
        X[...,ixs_lower[1],ixs_lower[0]] = x_vec[...,ndim:] / math.sqrt(2)
        X[...,ixs_diag,ixs_diag] = x_vec[...,:ndim]
        
        return X