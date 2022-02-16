import numpy as np
import numpy.linalg as la
from .ops import dense_ma
def rot_dense_to_sb(rot):
    L=int(np.log2(rot.shape[0]))
    ret=np.zeros((2*L,2*L))
    for i in range(2*L):
        for j in range(2*L):
            ret[i,j]=np.trace(dense_ma(L,j)@rot.T.conj()@dense_ma(L,i)@rot).real/2**L
    return ret

def rot_sb_to_dense(rot):
    raise NotImplementedError()

def rot_sb_to_sparse(rot):
    raise NotImplementedError()

def rot_sparse_to_sb(rot):
    raise NotImplementedError()
