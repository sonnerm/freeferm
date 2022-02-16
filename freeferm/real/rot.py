import numpy as np
import numpy.linalg as la
from .ops import dense_ma
from .. import eigu
def rot_dense_to_sb(rot):
    L=int(np.log2(rot.shape[0]))
    ret=np.zeros((2*L,2*L))
    for i in range(2*L):
        for j in range(2*L):
            ret[i,j]=np.trace(dense_ma(L,j)@rot.T.conj()@dense_ma(L,i)@rot).real/2**L
    return ret

def rot_sb_to_dense(rot):
    check_dense_lmax(rot.shape[0]//2)
    ev,evv=eigu(rot)
    eva=np.angle(ev) #equivalent to logarithgm
    evm,evvm=la.eigh(quad_sb_to_dense(evv.T.conj()@eva@evv))
    return evm@evvm


def rot_sb_to_sparse(rot):
    raise NotImplementedError()

def rot_sparse_to_sb(rot):
    raise NotImplementedError()
