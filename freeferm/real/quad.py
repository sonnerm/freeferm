import numpy as np
from .ops import dense_ma
from .. import dense_vac
def quad_sb_to_dense(quad):
    L=len(quad)//2
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            ret+=quad[i,j]*dense_ma(L,i)@dense_ma(L,j)
    return ret

def quad_sb_to_sparse(quad):
    return quad_sb_to_dense(quad) #correct, but maybe not efficient

def quad_sb_to_mpo(quad):
    raise NotImplementedError()

def quad_dense_to_sb(quad):
    L=int(np.log2(quad.shape[0]))
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            ret[i,j]=np.trace(dense_ma(L,i)@quad@dense_ma(L,j))/2**L
    ret-=np.diag(np.diag(ret))
    return ret*2

def quad_sparse_to_sb(quad):
    raise NotImplementedError()
