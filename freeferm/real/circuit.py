import numpy as np
from .. import block
from .rot import rot_sb_to_dense
import scipy.linalg as la
def rot_sb_to_circuit(rot):
    '''
        Decompose a single body rotation matrix into a quantum circuit.
    '''
    raise NotImplementedError()

def rot_circuit_to_sb(L,circ):
    ret=np.eye(2*L)
    for i,_,_,c in circ:
        ret=block([np.eye(2**(2*i)),c,np.eye(2**(2*L-2*i)//c.shape[0])])@ret
    return ret

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
    if la.det(mat)<0:
        h=np.array(mat[2])
        mat[2]=mat[3]
        mat[3]=h
    return mat

def _flamp_find_sb_gate(target):
    #start with a random matrix, set first row to target.real, second to target.imag
    #run Gram schmidt
    import flamp
    mat=np.random.random(size=(4,4))
    mat=flamp.to_mp(mat)
    tr=np.array([t.real for t in target])
    ti=np.array([t.imag for t in target])
    mat[0]=tr
    mat[0]/=flamp.sqrt(mat[0]@mat[0])
    mat[1]=ti
    mat[1]-=(mat[0]@mat[1])*mat[0]
    mat[1]/=flamp.sqrt(mat[1]@mat[1])
    mat[2]-=(mat[0]@mat[2])*mat[0]+(mat[1]@mat[2])*mat[1]
    mat[2]/=flamp.sqrt(mat[2]@mat[2])
    mat[3]-=(mat[0]@mat[3])*mat[0]+(mat[1]@mat[3])*mat[1]+(mat[2]@mat[3])*mat[2]
    mat[3]/=flamp.sqrt(mat[3]@mat[3])
    if flamp.det(mat)<0:
        h=mat[2].copy()
        mat[2]=mat[3]
        mat[3]=h
    return mat
def _flamp_corr_to_circuit(corr,nbcutoff=1e-10,prec=200):
    import flamp
    import gmpy2
    oldprec=flamp.get_precision()
    try:
        flamp.set_precision(prec)
        nbcutoff=gmpy2.mpfr(nbcutoff)
        ccorr=flamp.to_mp(corr)
        # ccorr=corr+0.5*np.eye(corr.shape[0])
        L=corr.shape[0]//2
        vs=[]
        for l in range(0,2*L,2):
            for b in range(2,2*L-l+1,2):
                sub=ccorr[l:b+l,l:b+l]
                ev,evv=flamp.eigh(sub)
                if max(ev)>gmpy2.mpfr(0.5)-nbcutoff:
                    target=evv[:,b-1]
                    break
                if b==2*L-l:
                    target=evv[:,b-1]
                    import warnings
                    warnings.warn("nbcutoff not reached %s"%str(0.5-max(ev)))
            for i in range(b-4,-1,-2):
                vs.append(((i+l)//2,_flamp_find_sb_gate(target[i:i+4])))
                _apply_rot_to_vec(target,i,vs[-1][1])
                _apply_rot_to_corr(ccorr,(i+l)//2,vs[-1][1])
        if ccorr[2*L-2,2*L-1].imag>0:
            for k,(i,r) in list(enumerate(vs))[::-1]:
                if i==L-2:
                    vs[k]=(i,flamp.to_mp(np.diag([1,1,1,-1]))@r)
                    break
        return [(v[0],rot_sb_to_dense(np.array(v[1],dtype=np.float64)).T.conj(),True if flamp.det(v[1])<0 else False,np.array(v[1].T,dtype=np.float64)) for v in vs[::-1]]
    finally:
        flamp.set_precision(oldprec)
    
def _corr_to_circuit(corr,nbcutoff=1e-10):
    ccorr=np.copy(corr)
    # ccorr=corr+0.5*np.eye(corr.shape[0])
    L=ccorr.shape[0]//2
    vs=[]
    for l in range(0,2*L,2):
        for b in range(2,2*L-l+1,2):
            sub=ccorr[l:b+l,l:b+l]
            try:
                ev,evv=la.eigh(sub)#Turns out that full diagonalization is in practice faster 
            except la.LinAlgError:
                import warnings
                warnings.warn("evr method did not converge, falling back to ev")
                ev,evv=la.eigh(sub,driver="ev")
            if max(ev)>0.5-nbcutoff:
                target=evv[:,-1]
                break
            if b==2*L-l:
                target=evv[:,-1]
                # assert False
                import warnings
                warnings.warn("nbcutoff not reached %f"%(0.5-max(ev)))
        for i in range(b-4,-1,-2):
            vs.append(((i+l)//2,_find_sb_gate(target[i:i+4])))
            _apply_rot_to_vec(target,i,vs[-1][1])
            _apply_rot_to_corr(ccorr,(i+l)//2,vs[-1][1])
    if ccorr[-2,-1].imag>0:
        for k,(i,r) in list(enumerate(vs))[::-1]:
            if i==L-2:
                vs[k]=(i,np.diag([1,1,1,-1])@r)
                break
    return [(v[0],rot_sb_to_dense(v[1]).T.conj(),True if la.det(v[1])<0 else False,v[1].T) for v in vs[::-1]]
def corr_to_circuit(corr,nbcutoff=1e-10,prec=None):
    '''
        Find a quantum circuit which transforms the vacuum state into the
        gaussian state with correlation matrix corr using a modified version of
        the algorithm described by Fishman and White Phys. Rev. B 92, 075132.
    '''
    if (nbcutoff>=1e-12 or prec is not None) and prec!=0:
        return _corr_to_circuit(corr,nbcutoff)
    elif prec is None:
        return _flamp_corr_to_circuit(corr,nbcutoff,prec=-int(np.log2(nbcutoff))*2)
    else:
        return _flamp_corr_to_circuit(corr,nbcutoff,prec)

def _apply_rot_to_corr(corr,pos,rot):
    corr[:2*pos,2*pos:2*pos+4]=corr[:2*pos,2*pos:2*pos+4]@rot.T
    corr[2*pos:2*pos+4,:2*pos]=rot@corr[2*pos:2*pos+4,:2*pos]
    corr[2*pos:2*pos+4,2*pos:2*pos+4]=rot@corr[2*pos:2*pos+4,2*pos:2*pos+4]@rot.T
    corr[2*pos:2*pos+4,2*pos+4:]=rot@corr[2*pos:2*pos+4,2*pos+4:]
    corr[2*pos+4:,2*pos:2*pos+4]=corr[2*pos+4:,2*pos:2*pos+4]@rot.T


def _apply_rot_to_vec(vec,i,rot):
    vec[i:i+4]=rot@vec[i:i+4]

def apply_circuit_to_corr(corr,circ):
    corr=corr.copy()
    for c in circ:
        _apply_rot_to_corr(corr,c[0],c[3])
    return corr
