import numpy as np
import numpy.linalg as la
from .. import check_sparse_lmax,apply_circuit_to_mps,mps_vac
from .quad import quad_sb_to_sparse,quad_sb_to_dense
from .ops import dense_ma
from .circuit import corr_to_circuit
from ttarray.raw import recluster,right_canonicalize
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
    #clustering ..., i think i'll just recluster to 2,2,2, ...
    counter=0
    for m in mps:
        counter+=int(math.log2(m.shape[0]))
    mps=recluster(mps,((2,),)*counter)
    right_canonicalize(mps)
    ret=np.zeros((len(mps)*2,len(mps)*2),dtype=complex)
    SX=np.array([[0,1],[1,0]])
    SY=np.array([[0,1j],[0,-1j]])
    SZ=np.array([[1,0],[0,-1]])
    ID=np.array([[1,0],[0,1]])
    bd=np.array([1.0])
    for i in range(len(mps)):
        bdsx=np.einsum("ab,bde,afg,df->eg",bd,mps[i],mps[i].conj(),SX)
        bdsy=np.einsum("ab,bde,afg,df->eg",bd,mps[i],mps[i].conj(),SY)
        for j in range(i,len(mps)):
            ret[2*i,2*j]=np.einsum("ab,bde,afg,df,eg",bdsx,mps[j],mps[j].conj(),SX,np.eye(mps[j].shape[-1]))
            ret[2*i,2*j+1]=np.einsum("ab,bde,afg,df,eg",bdsx,mps[j],mps[j].conj(),SY,np.eye(mps[j].shape[-1]))
            ret[2*i+1,2*j]=np.einsum("ab,bde,afg,df,eg",bdsy,mps[j],mps[j].conj(),SX,np.eye(mps[j].shape[-1]))
            ret[2*i+1,2*j+1]=np.einsum("ab,bde,afg,df,eg",bdsy,mps[j],mps[j].conj(),SY,np.eye(mps[j].shape[-1]))
            bdsx=np.einsum("ab,bde,afg,df->eg",bdsx,mps[i],mps[i].conj(),SZ)
            bdsy=np.einsum("ab,bde,afg,df->eg",bdsy,mps[i],mps[i].conj(),SZ)
        bd=np.einsum("ab,bde,afg,df->eg",bd,mps[i],mps[i].conj(),ID)
    ret-=ret.T
    return ret

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
