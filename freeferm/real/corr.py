import numpy as np
import math
import numpy.linalg as la
from .. import check_sparse_lmax,apply_circuit_to_mps,mps_vac
from .quad import quad_sb_to_sparse,quad_sb_to_dense
from .ops import dense_ma
from .circuit import corr_to_circuit
from ttarray.raw import recluster,right_canonicalize,left_canonicalize
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
def corr_to_mps(corr,cluster=None,nbcutoff=1e-10,chi=None,svd_cutoff=0.0):
    '''
        Find an MPS
    '''
    L=corr.shape[0]//2
    circ=corr_to_circuit(corr,nbcutoff)
    return apply_circuit_to_mps(mps_vac(L,cluster=cluster),circ,chi,svd_cutoff)

def mps_to_corr(mps,normalize=False):
    '''
        Calculate the correlation matrix of an MPS
    '''
    #clustering ..., i think i'll just recluster to 2,2,2, ...
    counter=int(math.log2(mps.shape[0]))
    mps=mps.recluster(((2,),)*counter)
    mps.canonicalize(0)
    mps=mps.asmatrices()
    ret=np.zeros((len(mps)*2,len(mps)*2),dtype=complex)
    SX=np.array([[0,1],[1,0]])
    SY=np.array([[0,1j],[-1j,0]])
    SZ=np.array([[1,0],[0,-1]])
    ID=np.array([[1,0],[0,1]])
    bd=np.array([[1.0]])
    norm=1.0
    for i in range(len(mps)):
        bdi=np.tensordot(bd,mps[i],axes=((0,),(0,)))
        bdi=np.tensordot(bdi,mps[i].conj(),axes=((0,),(0,)))

        bdsx=np.tensordot(bdi,SX,axes=((0,2),(0,1)))
        bdsy=np.tensordot(bdi,SY,axes=((0,2),(0,1)))
        if normalize:
            norm=np.trace(bd)
        ret[2*i,2*i]=0.0 # convention
        ret[2*i+1,2*i]=1.0j*np.trace(np.tensordot(bdi,SZ,axes=((0,2),(0,1))))
        for j in range(i+1,len(mps)):
            bdix=np.tensordot(bdsx,mps[j],axes=((0,),(0,)))
            bdix=np.tensordot(bdix,mps[j].conj(),axes=((0,),(0,)))
            bdiy=np.tensordot(bdsy,mps[j],axes=((0,),(0,)))
            bdiy=np.tensordot(bdiy,mps[j].conj(),axes=((0,),(0,)))
            ret[2*i,2*j]=1.0j*np.trace(np.tensordot(bdiy,SX,axes=((0,2),(0,1))))
            ret[2*i,2*j+1]=1.0j*np.trace(np.tensordot(bdiy,SY,axes=((0,2),(0,1))))
            ret[2*i+1,2*j]=-1.0j*np.trace(np.tensordot(bdix,SX,axes=((0,2),(0,1))))
            ret[2*i+1,2*j+1]=-1.0j*np.trace(np.tensordot(bdix,SY,axes=((0,2),(0,1))))
            bdsx=np.tensordot(bdix,SZ,axes=((0,2),(0,1)))
            bdsy=np.tensordot(bdiy,SZ,axes=((0,2),(0,1)))
        bd=np.tensordot(bdi,ID,axes=((0,2),(0,1)))
    ret-=ret.T
    return -ret/2/norm

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
