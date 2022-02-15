import numpy as np
from .utils import check_dense_lmax,check_sparse_lmax

def circuit_to_mps_uncompressed(init,circ):
    '''
        Apply a quantum circuit to an initial MPS without intermediate compression.
        Returns an MPS in the form of a list of matrices with the indices
    '''
    raise NotImplementedError()
def circuit_to_mps_svd(init,circ,chi,cutoff=1e-10):
    '''
        Apply a quantum circuit to an initial MPS and use svd to compress intermediate results.
        At each bond keep at most chi svd values which have at least weight cutoff.
    '''
    raise NotImplementedError()
def compress_svd(mps):
    '''
        Compresses an MPS given as a list of matrices with the indices inplace.
    '''
    raise NotImplementedError()

def mps_to_dense(mps):
    '''
        Converts a MPS to a dense
    '''
    check_sparse_lmax(len(mps))
    raise NotImplementedError()
def mpo_to_dense(mpo):
    check_dense_lmax(len(mpo))
    raise NotImplementedError()
