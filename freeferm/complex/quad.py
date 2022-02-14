import numpy as np
from .ops import *
def quad_sb_to_mb(quad):
    L=len(quad)
    ret=np.zeros((2**L,2**L),dtype=quad.dtype)
    for i in range(L):
        for j in range(L):
            ret+=quad[i,j]*cd(L,i)@c(L,j)
    return ret
def quad_mb_to_sb(quad):
    L=int(np.log2(len(quad)))
    ret=np.zeros((L,L),dtype=quad.dtype)
    vac=vac(L)
    vacs=[cd(L,i)@vac for i in range(L)]
    for i in range(L):
        for j in range(L):
            ret[i,j]=vacs[i].conj()@(quad@vacs[j])
    return ret
