from .. import kron,SX,SZ,SY,ID,check_dense_lmax,check_sparse_lmax
import functools
import numpy as np
@functools.lru_cache(None)
def dense_c(L,i):
    '''
        Dense $2^L\\times 2^L$ matrix representing the lowering (annihilation)
        operator for the fermion at site $i$ in a system of size $L$ commonly
        written as $\\hat{c}_i$. This function is lru_cached
    '''
    return dense_c_uncached(L,i)
def dense_c_uncached(L,i):
    '''
        Uncached version of dense_c
    '''
    check_dense_lmax(L)
    return kron([SZ]*i+[Sm]+[ID]*(L-i-1))
def sparse_c(L,i):
    raise NotImplementedError()

@functools.lru_cache(None)
def dense_cd(L,i):
    '''
        Dense $2^L\\times 2^L$ matrix representing the raising (creation)
        operator for the fermion at site $i$ in a system of size $L$ commonly
        written as $\\hat{c}^\\dagger_i$.
    '''
    return dense_cd_uncached(L,i)
def dense_cd_uncached(L,i):
    '''
        Uncached version of dense_cd
    '''
    check_dense_lmax(L)
    return kron([SZ]*i+[Sp]+[ID]*(L-i-1))

def sparse_cd(L,i):
    raise NotImplementedError()

@functools.lru_cache(None)
def dense_pn(L,n):
    '''
        Projection on the sector with particle number $n$. Commutes with any
        quadratic operator of complex fermions.
    '''
    return dense_pn_uncached(L,n)
def dense_pn_uncached(L,n):
    '''
        Uncached version of dense_pn
    '''
    check_dense_lmax(L)
    raise NotImplementedError()
def sparse_pn(L,n):
    raise NotImplementedError()


@functools.lru_cache(None)
def dense_n(L):
    return dense_n_uncached(L)
def dense_n_uncached(L):
    check_dense_lmax(L)
    raise NotImplementedError()
def sparse_n(L):
    raise NotImplementedError()
