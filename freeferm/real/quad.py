import numpy as np
from .ops import dense_ma
from ..complex import dense_c,dense_cd
from .. import dense_vac,dense_full
def quad_sb_to_dense(quad):
    L=len(quad)//2
    ret=np.zeros((2**L,2**L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            ret+=quad[i,j]*dense_ma(L,i)@dense_ma(L,j)
    return ret

def quad_sb_to_sparse(quad):
    raise NotImplementedError()
    # return quad_sb_to_dense(quad) #correct, but maybe not efficient

def quad_sb_to_mpo(quad):
    raise NotImplementedError()

def quad_dense_to_sb(quad):
    L=int(np.log2(quad.shape[0]))
    ret=np.zeros((2*L,2*L),dtype=complex)
    for i in range(2*L):
        for j in range(2*L):
            ret[i,j]=np.trace(dense_ma(L,i)@quad@dense_ma(L,j))/2**L
    ret-=np.diag(np.diag(ret))
    return ret*2

def quad_sparse_to_sb(quad):
    raise NotImplementedError()
    L=int(np.log2(len(quad)))
    ret=np.zeros((2*L,2*L),dtype=quad.dtype)
    vac=dense_vac(L)
    full=dense_full(L)
    qvac=quad@vac
    qfull=quad@full
    vacs=np.array([dense_cd(L,i)@vac for i in range(L)])
    fulls=np.array([dense_c(L,i)@full for i in range(L)])
    qvacs=np.array([quad@v for v in vacs])
    for i in range(L):
        for j in range(L):
            cc=(dense_cd(L,i)@vacs[j]).conj()@qvac
            dd=(dense_c(L,i)@fulls[j]).conj()@qfull
            cd=qvacs[i].conj()@qvacs[j]
            ret[2*i,2*j]=cc+dd+cd-cd.conj()
            ret[2*i+1,2*j]=1.0j*(cc-dd+cd+cd.conj())
            ret[2*i,2*j+1]=1.0j*(cc-dd-cd-cd.conj())
            ret[2*i+1,2*j+1]=cc+dd+cd.conj()-cd
    return np.einsum("ab,db",vacs.conj(),qvacs)
    # vac=dense_vac(L)/np.sqrt(2)+1.0j*dense_full(L)/np.sqrt(2)
    # vacs=[dense_ma(L,i)@vac for i in range(2*L)]
    # for i in range(2*L):
    #     for j in range(2*L):
    #         ret[i,j]=vacs[i].conj()@(quad@vacs[j])
    # return ret
