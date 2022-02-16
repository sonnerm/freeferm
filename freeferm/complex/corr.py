import numpy as np
from .. import check_sparse_lmax
from .quad import quad_sb_to_sparse,quad_sb_to_dense
from .ops import dense_c,dense_cd
def corr_to_dense(corr):
    '''
        Find the dense vector corresponding to the Gaussian state with correlation matrix corr.
    '''
    # TODO: implement sparse version
    check_sparse_lmax(corr.shape[0]//2)
    return la.eigh(quad_sb_to_dense(-corr+0.5*np.eye(corr.shape[0])))[1][:,0].T.conj()
def corr_to_mps(corr,nbcutoff=1e-10,chi=None,svd_cutoff=None):
    '''
        Find an MPS
    '''
    L=corr.shape[0]//2
    circ=corr_to_circuit(corr,nbcutoff)
    return circuit_to_mps(mps_vac(L),circ,chi,svd_cutoff)

def mps_to_corr(mps):
    '''
        Calculate the correlation matrix of an MPS
    '''
    raise NotImplementedError()
def dense_to_corr(dense):
    '''
        Calculate the correlation matrix of a dense vector
    '''
    L=int(np.log2(dense.shape[0]))
    corr=np.zeros((L,L),dtype=complex)
    phio=[dense_c(L,i)@dense for i in range(L)]
    for i in range(L):
        for j in range(L):
            corr[i,j]=phio[i].conj()@phio[j]
    return corr

def corr_vac(L):
    return np.zeros((L,L))
