import numpy as np
from .. import block
from .rot import rot_sb_to_dense
def rot_to_circuit(rot):
    '''
        Decompose a single body rotation matrix into a quantum circuit.
    '''
    raise NotImplementedError()
def _find_sb_gate(target):
    #start with a random matrix, set first row to target.real, second to target.imag
    #run Gram schmidt
    mat=np.random.random(size=(4,4))
    mat[0]=target.real
    mat[0]/=np.sqrt(mat[0]@mat[0])
    mat[1]=target.imag
    mat[1]-=(mat[0]@mat[1])*mat[0]
    mat[1]/=np.sqrt(mat[1]@mat[1])
    mat[2]-=(mat[0]@mat[2])*mat[0]+(mat[1]@mat[2])*mat[1]
    mat[2]/=np.sqrt(mat[2]@mat[2])
    mat[3]-=(mat[0]@mat[3])*mat[0]+(mat[1]@mat[3])*mat[1]+(mat[2]@mat[3])*mat[2]
    mat[3]/=np.sqrt(mat[3]@mat[3])
    return mat
def corr_to_circuit(corr,nbcutoff=1e-10):
    '''
        Find a quantum circuit which transforms the vacuum state into the
        gaussian state with correlation matrix corr using a modified version of
        the algorithm described by Fishman and White Phys. Rev. B 92, 075132.
    '''
    ccorr=np.copy(corr)
    L=ccorr.shape[0]//2
    vs=[]
    for l in range(0,2*L,2):
        for b in range(2,2*L-l+1,2):
            sub=ccorr[l:b+l,l:b+l]
            ev,evv=la.eigh(sub)
            if min(ev)<-1+nbcutoff:
                target=evv[:,0]
                break
        for i in range(b-4,-1,-2):
            vs.append((i+l,_find_sb_gate(target[i:i+4])))
            target=block([np.eye(i),vs[-1][1],np.eye(b-i-4)])@target
            rot=block([np.eye(i+l),vs[-1][1],2*L-i-l-4])
            ccorr=rot@ccorr@rot.T
    if ccorr[-2,-1].imag<0:
        for k,(i,r) in list(enumerate(vs))[::-1]:
            if i==2*L-4:
                vs[k]=(i,np.diag([1,1,1,-1])@r)
                break
    return [(v[0],rot_sb_to_dense(v[1])) for v in vs]
