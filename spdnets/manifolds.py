import torch
from typing import Union, Optional, Tuple
from geoopt.manifolds import Manifold
from . import functionals

__all__ = ["SymmetricPositiveDefinite"]

class SymmetricPositiveDefinite(Manifold):
    """
    Subclass of the SymmetricPositiveDefinite manifold using the 
    affine invariant Riemannian metric (AIRM) as default metric
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "SymmetricPositiveDefinite"
    ndim = 2
    reversible = False

    def __init__(self):
        super().__init__()

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim) -> torch.Tensor:
        """
        Computes the affine invariant Riemannian metric (AIM)
        """
        inv_sqrt_x = functionals.sym_invsqrtm.apply(x)
        return torch.norm(
            functionals.sym_logm.apply(inv_sqrt_x @ y @ inv_sqrt_x),
            dim=[-1, -2],
            keepdim=keepdim,
        )

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e = torch.linalg.eigvalsh(x)
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = functionals.ensure_sym(x)
        return functionals.sym_abseig.apply(symx)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return functionals.ensure_sym(u)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor], keepdim) -> torch.Tensor:
        if v is None:
            v = u
        inv_x = functionals.sym_invm.apply(x)
        ret = torch.diagonal(inv_x @ u @ inv_x @ v, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = functionals.sym_invm.apply(x)
        return functionals.ensure_sym(x + u + 0.5 * u @ inv_x @ u)
        # return self.expmap(x, u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        sqrt_x, inv_sqrt_x = functionals.sym_invsqrtm2.apply(x)
        return sqrt_x @ functionals.sym_expm.apply(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        sqrt_x, inv_sqrt_x = functionals.sym_invsqrtm2.apply(x)
        return sqrt_x @ functionals.sym_logm.apply(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def extra_repr(self) -> str:
        return "default_metric=AIM"
    
    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        xinvy = torch.linalg.solve(x.double(),y.double())
        s, U = torch.linalg.eig(xinvy.transpose(-2,-1))
        s = s.real
        U = U.real

        Ut = U.transpose(-2,-1)
        Esqm = torch.linalg.solve(Ut, torch.diag_embed(s.sqrt()) @ Ut).transpose(-2,-1).to(y.dtype)

        return Esqm @ v @ Esqm.transpose(-1,-2)

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        tens = torch.randn(*size, dtype=dtype, device=device, **kwargs)
        tens = functionals.ensure_sym(tens)
        tens = functionals.sym_expm.apply(tens)
        return tens

    def barycenter(self, X : torch.Tensor, steps : int = 1, dim = 0) -> torch.Tensor:
        """
        Compute several steps of the Kracher flow algorithm to estimate the 
        Barycenter on the manifold.
        """
        return functionals.spd_mean_kracher_flow(X, None, maxiter=steps, dim=dim, return_dist=False)

    def geodesic(self, A : torch.Tensor, B : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic between two SPD tensors A and B and return 
        point on the geodesic at length t \in [0,1]
        if t = 0, then A is returned
        if t = 1, then B is returned
        """
        Asq, Ainvsq = functionals.sym_invsqrtm2.apply(A)
        return Asq @ functionals.sym_powm.apply(Ainvsq @ B @ Ainvsq, t) @ Asq

    def transp_via_identity(self, X : torch.Tensor, A : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Parallel transport from around the identity matrix to the new center (tensor B)
        """
        Ainvsq = functionals.sym_invsqrtm.apply(A)
        Bsq = functionals.sym_sqrtm.apply(B)
        return Bsq @ (Ainvsq @ X @ Ainvsq) @ Bsq

    def transp_identity_rescale_transp(self, X : torch.Tensor, A : torch.Tensor, s : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Rescales the dispersion by the factor s
        Parallel transport from the identity to the new center (tensor B)
        """
        Ainvsq = functionals.sym_invsqrtm.apply(A)
        Bsq = functionals.sym_sqrtm.apply(B)
        return Bsq @ functionals.sym_powm.apply(Ainvsq @ X @ Ainvsq, s) @ Bsq

    def transp_identity_rescale_rotate_transp(self, X : torch.Tensor, A : torch.Tensor, s : torch.Tensor, B : torch.Tensor, W : torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Rescales the dispersion by the factor s
        Parallel transport from the identity to the new center (tensor B)
        """
        Ainvsq = functionals.sym_invsqrtm.apply(A)
        Bsq = functionals.sym_sqrtm.apply(B)
        WBsq = W @ Bsq
        return WBsq.transpose(-2,-1) @ functionals.sym_powm.apply(Ainvsq @ X @ Ainvsq, s) @ WBsq
