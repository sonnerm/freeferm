import numpy as np
def rot_to_circuit(rot):
    '''
        Decompose a single body rotation matrix into a quantum circuit.
    '''
    raise NotImplementedError()
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
            vs.append((i+l,single_body_gate(target[i:i+4])))
            target=single_body_rotation(b,i,vs[-1][1])@target
            rot=single_body_rotation(2*L,vs[-1][0],vs[-1][1])
            ccorr=rot@ccorr@rot.T
    if ccorr[-2,-1].imag<0:
        for k,(i,r) in list(enumerate(vs))[::-1]:
            if i==2*L-4:
                vs[k]=(i,np.diag([1,1,1,-1])@r)
                break
