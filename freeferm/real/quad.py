import numpy as np
from .ops import ma
def quad_sb_to_dense(quad):
    L=len(quad)//2
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            ret+=quad[i,j]*ma(L,i)@ma(L,j)
    return ret

def quad_sb_to_sparse(quad):
    return quad_sb_to_dense(quad) #correct, but maybe not efficient

def quad_sb_to_mpo(quad):
    raise NotImplementedError()

def quad_dense_to_sb(quad):
    return quad_sparse_to_sb(quad) #correct, but maybe not efficient

def quad_sparse_to_sb(quad):
    raise NotImplementedError()
