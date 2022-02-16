import numpy as np
import numpy.linalg as la
from .utils import check_dense_lmax,check_sparse_lmax,SZ

def circuit_to_mps(init,circ,chi=None,cutoff=None):
    '''
        Apply a quantum circuit to an initial MPS inplace. If chi or cutoff is set,
        it performs an svd compression to limit bond dimension or cutoff low degrees
        of freedom. For convenience it returns init, an MPS in the form of a list of
        matrices with the indices (left,physical,right)
    '''
    for c in circ:
        i,gate,stri=c[0],c[1],c[2]
        if stri:
            for j in range(i):
                init[j]=np.einsum("abc,bd->adc",init[j],SZ)
        l=int(np.log2(gate.shape[0]))
        inter=dense_to_mps_slice(np.einsum("abc,bd->adc",mps_slice_to_dense(init[i:i+l]),gate))
        if chi is not None or cutoff is not None:
            inter=compress_svd(inter,chi,cutoff)
        init[i:i+l]=inter
    return init
def compress_svd(mps,chi=None,cutoff=None):
    '''
        Compresses an MPS given as a list of matrices with the indices
        (left,physical,right) inplace. Returns mps for convenience.
    '''
    raise NotImplementedError()
def mps_slice_to_dense(mps):
    check_sparse_lmax(len(mps))
    ret=mps[0]
    for m in mps[1:]:
        ret=np.einsum("abc,cde->abde",ret,m).reshape((ret.shape[0],ret.shape[1]*m.shape[1],m.shape[2]))
    return ret
def mps_to_dense(mps):
    '''
        Converts a MPS to a dense vector
    '''
    return mps_slice_to_dense(mps)[0,:,0]

def dense_to_mps(dense):
    '''
        Converts a dense vector to a canonicalized MPS
    '''
    return dense_to_mps_slice(dense.reshape((1,dense.shape[0],1)))
def dense_to_mps_slice(dense):
    mps=[]
    L=int(np.log2(dense.shape[1]))
    cdense=dense.reshape(dense.shape[0],dense.shape[1]*dense.shape[2])
    for i in range(L):
        cdense=cdense.reshape((cdense.shape[0]*2,(cdense.shape[1])//2))
        q,r=la.qr(cdense)
        mps.append(q.reshape((q.shape[0]//2,2,q.shape[1])))
        cdense=r
    mps[-1]*=r
    return mps

def mpo_slice_to_dense(mpo):
    check_dense_lmax(len(mpo))
    ret=mpo[0]
    for m in mps[1:]:
        ret=np.einsum("abcd,defg->abecfg",ret,m).reshape((ret.shape[0],ret.shape[1]*m.shape[1],ret.shape[2]*m.shape[2],m.shape[3]))
    return ret
def mpo_to_dense(mpo):
    return mpo_slice_to_dense(mpo)[0,:,:,0]
def dense_to_mpo(dense):
    return dense_to_mpo_slice(dense.reshape((1,dense.shape[0],dense.shape[1],1)))
def dense_to_mpo_slice(dense):
    raise NotImplementedError()
def mps_vac(L):
    return [np.array([0,1]).reshape((1,2,1))]*L

def mps_full(L):
    return [np.array([1,0]).reshape((1,2,1))]*L
