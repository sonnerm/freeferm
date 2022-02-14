from .. import kron,SX,SZ,SY,ID
import functools
import numpy as np
@functools.lru_cache(None)
def me(L,i):
    '''
        Even Majorana operator at site $i$ in a system with size $L$. The
        convention is $\\hat{\\gamma}_{2i} = \\hat{c}_i+\\hat{c}^\\dagger_i$.
        This function is lru_cached.
    '''
    return kron([SZ]*i+[SX]+[ID]*(L-i-1))
@functools.lru_cache(None)
def mo(L,i):
    '''
        Odd Majorana operator at site $i$ in a system with size $L$. The
        convention is $\\hat{\\gamma}_{2i+1} = 1j(\\hat{c}^\\dagger_i-\\hat{c}_i)$.
        This function is lru_cached.
    '''
    return kron([SZ]*i+[SY]+[ID]*(L-i-1))

def ma(L,i):
    if i%2==1:
        return mo(L,i)
    else:
        return me(L,i)

@functools.lru_cache(None)
def pe(L):
    return (np.eye(2**L)+kron([SZ]*L))//2
@functools.lru_cache(None)
def po(L):
    '''
        Projection on the odd parity sector, commutes with any quadratic
        operator of real fermions.
    '''
    return (np.eye(2**L)-kron([SZ]*L))//2
