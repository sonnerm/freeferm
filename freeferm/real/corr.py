import numpy as np
import numpy.linalg as la
from .. import check_sparse_lmax,apply_circuit_to_mps,mps_vac
from .quad import quad_sb_to_sparse,quad_sb_to_dense
from .ops import dense_ma
from .circuit import corr_to_circuit
def corr_to_dense(corr,sparse=False):
    '''
        Find the dense vector corresponding to the Gaussian state with correlation matrix corr.
    '''
    check_sparse_lmax(corr.shape[0]//2)
    if sparse:
        import scipy.sparse.linalg as sla
        sla.eigsh(quad_sb_to_sparse(corr),k=1,which="SA")[1][:,0]
    else:
        return la.eigh(quad_sb_to_dense(corr))[1][:,0]
def corr_to_mps(corr,cluster=None,nbcutoff=1e-10,chi=None,svd_cutoff=None):
    '''
        Find an MPS
    '''
    L=corr.shape[0]//2
    circ=corr_to_circuit(corr,nbcutoff)
    return apply_circuit_to_mps(mps_vac(L,cluster=cluster),circ,chi,svd_cutoff)

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
    phio=[dense_ma(L,i)@dense for i in range(2*L)]
    for i in range(2*L):
        for j in range(2*L):
            corr[i,j]=phio[i].conj()@phio[j]
    return corr-np.eye(2*L)/2 #our convention for the correlation matrix
def corr_vac(L):
    return np.diag(([-0.5j,0]*L)[:-1],1)+np.diag(([0.5j,0]*L)[:-1],-1)

def corr_full(L):
    return np.diag(([0.5j,0]*L)[:-1],1)+np.diag(([-0.5j,0]*L)[:-1],-1)
