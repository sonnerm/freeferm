import numpy as np
import numpy.linalg as la
def rot_sb_to_circuit(rot):
    '''
        Decompose a single body rotation matrix into a quantum circuit.
    '''
    raise NotImplementedError()

def rot_circuit_to_sb(rot):
    ret=np.eye(L)
    for i,_,_,c in circ:
        ret=block([np.eye(2**i),c,np.eye(2**(L-i)//c.shape[0])])@ret
    return ret
def _find_sb_gate(target):
    mat=np.zeros((4,4),dtype=complex)
    mat[0]=target
    mat[0]=mat[0]/np.sqrt(mat[0]@mat[0])
    mat[1,0]=mat[0,1]
    mat[1,1]=-mat[0,0]
    return mat

def corr_to_circuit(corr,nbcutoff=1e-10):
    '''
        Find a quantum circuit which transforms the vacuum state into the
        gaussian state with correlation matrix corr using the algorithm
        described by Fishman and White Phys. Rev. B 92, 075132. Quantum circuits
        are list of the form (index, many body gate, Jordan-Wigner string
        necessary?, single body gate).
    '''
    ccorr=np.copy(corr)
    L=ccorr.shape[0]
    vs=[]
    init=[]
    for l in range(L):
        for b in range(1,L-l+1):
            sub=ccorr[l:b+l,l:b+l]
            ev,evv=la.eigh(sub)
            if min(ev)<nbcutoff:
                init.append(0)
                target=evv[:,0]
                break
            elif max(ev)>1-nbcutoff:
                init.append(1)
                target=evv[:,-1]
                break
        for i in range(b-2,-1,-1):
            vs.append((i+l,_find_sb_gate(target[i:i+2])))
            target=single_body_rotation(b,i,vs[-1][1],vs[-1][2]).T.conj()@target
            rot=single_body_rotation(L,vs[-1][0],vs[-1][1],vs[-1][2])
            ccorr=rot.T.conj()@ccorr@rot
    return vs
