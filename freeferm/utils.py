import numpy as np
SX=np.array([[0,1],[1,0]])
SY=np.array([[0,1.0j],[-1.0j,0]])
SZ=np.array([[1,0],[0,-1]])
ID=np.array([[1,0],[0,1]])
Sp=np.array([[0,0],[1,0]])
Sm=np.array([[0,1],[0,0]])
LMAX=8
def check_lmax(L):
    '''
        Ensure that we do not run out of memory by using excessive system sizes.
        Configurable through the constant LMAX. Remember that memory of states
        scales as 2**L and memory of operators scales as 4**L.
    '''
    if L>LMAX:
        raise ValueError("System size is too large (%i>%i). Either use smaller system size or increase maximal system size LMAX."%(L,LMAX))
def kron(args):
    '''
        Calculate the kronecker product (outer product) of a list of matrices
    '''
    ret=args[0]
    for a in args[1:]:
        ret=np.kron(a,ret)
    return ret

def outer(args):
    '''
        Calculate the outer product of a list of vectors
    '''
    ret=args[0]
    for a in args[1:]:
        ret=np.outer(a,ret)
    return ret
