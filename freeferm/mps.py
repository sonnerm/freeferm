import numpy as np
import math
from .utils import SZ,ID
import ttarray as tt

def locate_tensor(i,cluster):
    for j,c in enumerate(cluster):
        i=i//c
        if i<1:
            return j
    return len(cluster)-1

def convert_circuit_to_mpo(L,circ):
    circ=list(circ)
    res=[]
    while circ:
        touched=set()
        cmpo=[]
        ncirc=[]
        for c in circ:
            if c[0] not in touched and c[0]+1 not in touched and not c[2]:
                touched.add(c[0])
                touched.add(c[0]+1)
                cmpo.extend([np.eye(2).reshape((1,2,2,1)) for _ in range(c[0]-len(cmpo))])
                cmpo.extend(tt.array(c[1]).tomatrices_unchecked())
            elif c[2]:
                for i in range(c[0]+2):
                    if i in touched:
                        ncirc.append(c)
                        break
                else:
                    touched.update(range(c[0]+2))
                    cmpo.extend([SZ.reshape((1,2,2,1)) for _ in range(c[0])])
                    cmpo.extend(tt.array(c[1]).tomatrices_unchecked())
            else:
                ncirc.append(c)
        circ=ncirc
        cmpo.extend([np.eye(2).reshape((1,2,2,1)) for _ in range(L-len(cmpo))])
        res.append(tt.frommatrices(cmpo))
    return res


def apply_circuit_to_mps(init,circ,chi=None,cutoff=1e-18):
    '''
        Apply a quantum circuit to an initial MPS inplace. If chi or cutoff is set,
        it performs an svd compression to limit bond dimension or cutoff low degrees
        of freedom. For convenience it returns init, an MPS in the form of a list of
        matrices with the indices (left,physical,right)
    '''
    for c in circ:
        i,gate,stri=c[0],c[1],c[2]
        if stri:
            mpo=tt.fromproduct([SZ]*i+[gate]+[ID]*(int(math.log2(init.shape[0]//2**i//gate.shape[0]))))
            mpo.recluster([(i[0],i[0]) for i in init.cluster])
        else:
            mpo=tt.fromproduct([ID]*i+[gate]+[ID]*(int(math.log2(init.shape[0]//2**i//gate.shape[0]))))
            mpo.recluster([(i[0],i[0]) for i in init.cluster])
        init.canonicalize(locate_tensor(2**i*gate.shape[1],[cl[0] for cl in init.cluster]))
        ncenter=init.center
        nncenter=locate_tensor(2**i,[cl[0] for cl in init.cluster])
        init=mpo@init
        init.setcenter_unchecked(ncenter) # this is actually not true, but it will work
        init.canonicalize(locate_tensor(2**i,[cl[0] for cl in init.cluster]))
        if chi is not None or cutoff is not None:
            init.truncate(left=nncenter,right=ncenter,chi_max=chi,cutoff=cutoff)
    return init

# how it should look like with ttarray finished and smart: (compare to apply_circuit_to_dense...)
# def apply_circuit_to_mps_2(init,circ):
#     for c in circ:
#         i,gate,stri=c[0],c[1],c[2]
#         if stri:
#             mpo=ttarray.mouter([SZ]*i+[gate]+[ID]*(int(math.log2(init.shape[0]//2**i//gate.shape[0]))))
#         else:
#             mpo=ttarray.mouter([ID]*i+[gate]+[ID]*(int(math.log2(init.shape[0]//2**i//gate.shape[0]))))
#         init=mpo@init
#         # truncation params should be set globally i think, (or in init)
#     return init
def mps_vac(L,cluster=None):
    ret=tt.fromproduct([np.array([0,1])]*L)
    if cluster is None:
        return ret
    ret.recluster(cluster)
    return ret

def mps_full(L,cluster=None):
    ret=tt.fromproduct([np.array([1,0])]*L)
    if cluster is None:
        return ret
    ret.recluster(ret,cluster)
    return ret
