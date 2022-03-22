import numpy as np
import math
import numpy.linalg as la
from .utils import check_dense_lmax,check_sparse_lmax,SZ,ID,kron
from ttarray.raw import recluster,left_truncate_svd,tensordot,shift_orthogonality_center
from ttarray.raw import left_canonicalize,is_canonical
from functools import reduce

def locate_tensor(i,cluster):
    for j,c in enumerate(cluster):
        i=i//c
        if i<1:
            return j
    return len(cluster)-1
def apply_circuit_to_mps(init,circ,chi=None,cutoff=None):
    '''
        Apply a quantum circuit to an initial MPS inplace. If chi or cutoff is set,
        it performs an svd compression to limit bond dimension or cutoff low degrees
        of freedom. For convenience it returns init, an MPS in the form of a list of
        matrices with the indices (left,physical,right)
    '''
    mps_sz=SZ[None,...,None]
    mps_id=ID[None,...,None]
    ishape=reduce(lambda x,y:x*y, (x.shape[1] for x in init))
    cluster=tuple([(x.shape[1],x.shape[1]) for x in init])
    left_canonicalize(init)
    center=len(init)-1
    for c in circ:
        i,gate,stri=c[0],c[1],c[2]
        mps_gate=gate[None,...,None]
        if stri:
            tts=[mps_sz]*i+[mps_gate]+[mps_id]*(int(math.log2(ishape//2**i//gate.shape[1])))
        else:
            tts=[mps_id]*i+[mps_gate]+[mps_id]*(int(math.log2(ishape//2**i//gate.shape[1])))
        tts=recluster(tts, cluster)
        ncenter=locate_tensor(2**i*gate.shape[1],[cl[0] for cl in cluster])
        nncenter=locate_tensor(2**i,[cl[0] for cl in cluster])
        shift_orthogonality_center(init,center,ncenter)
        center=ncenter
        assert is_canonical(init,center)
        init=tensordot(tts,init,((1,),(0,)))
        if chi is not None or cutoff is not None:
            initsep=init[nncenter:ncenter+1]
            left_truncate_svd(initsep,chi_max=chi,cutoff=cutoff)
            init[nncenter:ncenter+1]=initsep
            center=nncenter
            assert is_canonical(init,center)
        else:
            # still make sure we don't exceed the physical maximum
            center=nncenter
            shift_orthogonality_center(init,ncenter,center)
            assert is_canonical(init,center)
    return init

# how it should look like with ttarray finished: (compare to apply_circuit_to_dense...)
# def apply_circuit_to_mps_2(init,circ):
#     for c in circ:
#         i,gate,stri=c[0],c[1],c[2]
#         if stri:
#             mpo=ttarray.mouter([SZ]*i+[gate]+[ID]*(int(math.log2(init.shape[0]//2**i//gate.shape[0]))))
#         else:
#             mpo=ttarray.mouter([ID]*i+[gate]+[ID]*(int(math.log2(init.shape[0]//2**i//gate.shape[0]))))
#         init=mpo@init
#         # truncation params should be set globally i think, otherwise:
#         # init.truncate(chi=chi,cutoff=cutoff)
#     return init
def mps_vac(L):
    return [np.array([0,1]).reshape((1,2,1))]*L

def mps_full(L):
    return [np.array([1,0]).reshape((1,2,1))]*L
