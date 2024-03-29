import numpy as np
import numpy.linalg as la
from .ops import dense_ma
from .quad import quad_sb_to_dense
from .. import eigu,check_dense_lmax,check_sparse_lmax,kron,SX,ID,SZ
def rot_dense_to_sb(rot):
    L=int(np.log2(rot.shape[0]))
    ret=np.zeros((2*L,2*L))
    for i in range(2*L):
        for j in range(2*L):
            ret[i,j]=np.trace(dense_ma(L,j)@rot.T.conj()@dense_ma(L,i)@rot).real/2**(L-1)
    return ret
def rot_sb_to_dense(rot):
    #To do properly: 
    #a) sort the eigenvalues in complex conjugate pairs
    #b) check if 
    L=rot.shape[0]//2
    check_dense_lmax(L)
    pm=np.eye(2**L)
    arot=rot
    if la.det(arot)<0:
        arot=rot@np.diag([-1]+[1]*(2*L-1))
        pm=kron([SX]+[ID]*(L-1))
    if np.trace(arot)<0:
        arot=-arot
        pm=pm@kron([SZ]*L)
    ev,evv=eigu(arot)
    eva=np.angle(ev) #equivalent to matrix logarithm for unit values
    evm,evvm=la.eigh(quad_sb_to_dense(np.einsum("ab,b,cb->ac",evv,eva,evv.conj())))
    ret=np.einsum("ab,b,cb->ac",evvm,np.exp(0.5j*evm),evvm.conj())@pm
    return ret
    rret=rot_dense_to_sb(ret@pm)
    # if np.allclose(rret,rot): #This fixes the ambiguity issue for L=2
    #     return ret
    # elif np.allclose(rret,-rot):
    #     return ret@kron([SZ]*L)
    # else:
    #     raise la.LinAlgError("Rotation did not work",rret,rot,rret-rot)


def rot_sb_to_sparse(rot):
    raise NotImplementedError()
def rot_sparse_to_sb(rot):
    raise NotImplementedError()
