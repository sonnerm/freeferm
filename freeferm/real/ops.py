from .. import kron,SX,SZ,SY,ID,check_dense_lmax
import functools
import numpy as np
@functools.lru_cache(None)
def dense_me(L,i):
    '''
        Dense representation of the even Majorana operator at site $i$ in a
        system with size $L$. The convention is $\\hat{\\gamma}_{2i}
        =\\frac{1}{\\sqrt{2}}( \\hat{c}_i+\\hat{c}^\\dagger_i)$. This function
        is lru_cached.
    '''
    return dense_me_uncached(L,i)


def dense_me_uncached(L,i):
    '''
        Uncached version of dense_me(L,i)
    '''
    check_dense_lmax(L)
    return kron([SZ]*i+[SX/np.sqrt(2)]+[ID]*(L-i-1))

def sparse_me(L,i):
    raise NotImplementedError()
@functools.lru_cache(None)
def dense_mo(L,i):
    '''
        Dense representation of the odd Majorana operator at site $i$ in a
        system with size $L$. The convention is $\\hat{\\gamma}_{2i+1} =
        \\frac{1j}{\\sqrt{2}}(\\hat{c}^\\dagger_i-\\hat{c}_i)$. This function is
        lru_cached.
    '''
    return dense_mo_uncached(L,i)

def dense_mo_uncached(L,i):
    '''
        Uncached version of dense_mo
    '''
    check_dense_lmax(L)
    return kron([SZ]*i+[SY/np.sqrt(2)]+[ID]*(L-i-1))

def sparse_mo(L,i):
    raise NotImplementedError()
def dense_ma(L,i):
    if i%2==1:
        return dense_mo(L,i//2)
    else:
        return dense_me(L,i//2)

def sparse_ma(L,i):
    if i%2==1:
        return sparse_mo(L,i//2)
    else:
        return sparse_me(L,i//2)

@functools.lru_cache(None)
def dense_pe(L):
    '''
        Projection onto the even parity sector, commutes with any quadratic
        operator of real fermions. This function is lru_cached
    '''
    return dense_pe_uncached(L)

def dense_pe_uncached(L):
    '''
        Uncached version of dense_pe
    '''
    check_dense_lmax(L)
    return (np.eye(2**L)+kron([SZ]*L))//2

def sparse_pe(L):
    raise NotImplementedError()

@functools.lru_cache(None)
def dense_po(L):
    '''
        Projection on the odd parity sector, commutes with any quadratic
        operator of real fermions.
    '''
    return dense_po_uncached(L)

def dense_po_uncached(L):
    '''
        Uncached version of dense_pe
    '''
    check_dense_lmax(L)
    return (np.eye(2**L)-kron([SZ]*L))//2
def sparse_po(L):
    raise NotImplementedError()
