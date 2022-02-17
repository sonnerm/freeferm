import numpy as np
from .ops import *
from .. import dense_vac
def quad_sb_to_dense(quad):
    '''
        Convert the single body matrix representation of a quadratic operator to a dense many-body form
    '''
    L=len(quad)
    ret=np.zeros((2**L,2**L),dtype=quad.dtype)
    for i in range(L):
        for j in range(L):
            ret+=quad[i,j]*dense_cd(L,i)@dense_c(L,j)
    return ret

def quad_sb_to_sparse(quad):
    '''
        Convert the single body matrix representation of a quadratic fermionic operator to a sparse linear operator
    '''
    raise NotImplementedError()
def quad_sb_to_mpo(quad):
    raise NotImplementedError() #coming soon
def quad_dense_to_sb(quad):
    '''
        Find the single body matrix representation of the dense representation of a quadratic operator
    '''
    return quad_sparse_to_sb(quad)

def quad_sparse_to_sb(quad):
    '''
        Find the single body matrix representation of a sparse linear operator representation of a quadratic operator
    '''
    L=int(np.log2(len(quad)))
    ret=np.zeros((L,L),dtype=quad.dtype)
    vac=dense_vac(L)
    vacs=np.array([dense_cd(L,i)@vac for i in range(L)])
    qvacs=np.array([quad@v for v in vacs])
    return np.einsum("ab,db",vacs.conj(),qvacs)
