import numpy as np
import numpy.linalg as la
from .utils import check_dense_lmax,check_sparse_lmax

def circuit_to_mps_uncompressed(init,circ):
    '''
        Apply a quantum circuit to an initial MPS without intermediate
        compression. Returns an MPS in the form of a list of matrices with the
        indices (left,right,physical)
    '''
    for c in circ:
        i,gate,stri=c[0],c[1],c[2]
    raise NotImplementedError()
def circuit_to_mps_svd(init,circ,chi,cutoff=1e-10):
    '''
        Apply a quantum circuit to an initial MPS and use svd to compress
        intermediate results. At each bond keep at most chi svd values which
        have at least weight cutoff.
    '''
    raise NotImplementedError()
def compress_svd(mps):
    '''
        Compresses an MPS given as a list of matrices with the indices
        (left,right,physical) inplace.
    '''
    raise NotImplementedError()

def mps_to_dense(mps):
    '''
        Converts a MPS to a dense vector
    '''
    check_sparse_lmax(len(mps))
    ret=mps[0][0].T
    for m in mps[1:]:
        ret=np.einsum("ab,bcd->adc",ret,m).reshape((ret.shape[0]*m.shape[2],m.shape[1]))
    return ret[:,0]

def dense_to_mps(dense):
    '''
        Converts a dense vector to a canonicalized MPS
    '''
    mps=[]
    ret
    raise NotImplementedError()

def mpo_to_dense(mpo):
    check_dense_lmax(len(mpo))
    ret=mpo[0][0].transpose([1,2,0])
    for m in mps[1:]:
        ret=np.einsum("abc,cdef->aebfd",ret,m).reshape((ret.shape[0]*m.shape[2],ret.shape[1]*m.shape[3],m.shape[1]))
    return ret[:,:,0]

def dense_to_mpo(mpo):
    raise NotImplementedError()
def mps_vac(L):
    raise NotImplementedError()
