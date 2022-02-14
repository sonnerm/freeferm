from .. import kron,SX,SZ,SY,ID,check_lmax
import functools
import numpy as np
@functools.lru_cache(None)
def c(L,i):
    '''
        Dense $2^L\\times 2^L$ matrix representing the lowering (annihilation)
        operator for the fermion at site $i$ in a system of size $L$ commonly
        written as $\\hat{c}_i$.
    '''
    check_lmax(L)
    return kron([SZ]*i+[Sm]+[ID]*(L-i-1))
@functools.lru_cache(None)
def cd(L,i):
    '''
        Dense $2^L\\times 2^L$ matrix representing the raising (creation)
        operator for the fermion at site $i$ in a system of size $L$ commonly
        written as $\\hat{c}^\\dagger_i$.
    '''
    check_lmax(L)
    return kron([SZ]*i+[Sp]+[ID]*(L-i-1))
@functools.lru_cache(None)
def pn(L,n):
    '''
        Projection on the sector with particle number $n$. Commutes with any
        quadratic operator of complex fermions.
    '''
    check_lmax(L)
    pass
@functools.lru_cache(None)
def n(L):
    pass
@functools.lru_cache(None)
def vac(L):
    check_lmax(L)
    ret=np.zeros((2**L,))
    ret[0]=1
    return ret
