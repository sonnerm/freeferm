import numpy as np
from .. import check_sparse_lmax
def corr_to_dense(corr):
    '''
        Find the dense vector corresponding to the Gaussian state with correlation matrix corr.
    '''
    check_sparse_lmax(corr.shape[0]//2)
    raise NotImplementedError()
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
    raise NotImplementedError()
