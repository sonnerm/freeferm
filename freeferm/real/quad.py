import numpy as np
def quad_sb_to_mb(quad):
    L=len(quad)//2
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            ret+=quad[i,j]*ma(L,i)@ma(L,j)
    return ret

def quad_mb_to_sb(quad):
    pass
