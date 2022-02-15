import numpy as np
import numpy.linalg as la
from .. import check_sparse_lmax
from .quad import quad_sb_to_sparse
def corr_to_dense(corr,sparse=False):
    '''
        Find the dense vector corresponding to the Gaussian state with correlation matrix corr.
    '''
    check_sparse_lmax(corr.shape[0]//2)
    if sparse:
        import scipy.sparse.linalg as sla
        sla.eigsh(quad_sb_to_sparse(corr),k=1,which="SA")[1][:,0]
    else:
        return la.eigh(quad_sb_to_dense(corr))[1][:,0] #for now maybe later change algo to fishman
def corr_to_mps(corr,nbcutoff=1e-10,chi=None,svd_cutoff=None):
    '''
        Find an MPS
    '''
    L=corr.shape[0]//2
    circ=corr_to_circuit(corr,nbcutoff)
    if chi is None and svd_cutoff is None:
        return circuit_to_mps_uncompressed(mps_vac(L),circ)
    else:
        return circuit_to_mps_svd(mps_vac(L),circ,chi,svd_cutoff)

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
    corr=np.zeros((2*L,2*L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            corr[i,j]=dense.conj()@(ma(L,i)@(ma(L,j)@dense))
    return corr-np.eye(2*L) #our convention for the correlation matrix
def corr_vac(L):
    return np.diag(([1.0j,0]*L)[:-1],1)+np.diag(([-1.0j,0]*L)[:-1],-1)
