import numpy as np
SX=np.array([[0,1],[1,0]])
SY=np.array([[0,-1.0j],[1.0j,0]])
SZ=np.array([[1,0],[0,-1]])
ID=np.array([[1,0],[0,1]])
Sp=np.array([[0,1],[0,0]])
Sm=np.array([[0,0],[1,0]])
DENSE_LMAX=8
SPARSE_LMAX=16
def check_dense_lmax(L):
    '''
        Ensure that we do not run out of memory by using excessive system sizes.
        Configurable through the constant DENSE_LMAX. Remember that memory of dense
        operators scales as 4**L.
    '''
    if L>DENSE_LMAX:
        raise ValueError("System size is too large (%i>%i). Either use smaller system size or increase maximal system size DENSE_LMAX."%(L,DENSE_LMAX))

def check_sparse_lmax(L):
    '''
        Ensure that we do not run out of memory by using excessive system sizes.
        Configurable through the constant LMAX. Remember that memory of vectors
        necessary for sparse linear algebra still scales as 2**L.
    '''
    if L>SPARSE_LMAX:
        raise ValueError("System size is too large (%i>%i). Either use smaller system size or increase maximal system size SPARSE_LMAX."%(L,SPARSE_LMAX))
def kron(args):
    '''
        Calculate the kronecker product (=outer product for matrices) of a list of matrices
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
    return ret.ravel()
def dense_vac(L):
    '''
        Returns a dense vector representing the fermionic vacuum state (i.e. no particle state)
    '''
    check_sparse_lmax(L)
    ret=np.zeros((2**L,))
    ret[-1]=1
    return ret
