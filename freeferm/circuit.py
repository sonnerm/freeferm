import numpy as np
from .utils import kron,SZ,ID
def apply_circuit_to_dense(init,circuit):
    '''
        Apply a quantum circuit to a dense vector
    '''
    L=int(np.log2(init.shape[0]))
    for c in circuit:
        i,gate,stri=c[0],c[1],c[2]
        l=int(np.log2(gate.shape[0]))
        if stri:
            op=kron([SZ]*i+[gate]+[ID]*(L-i-l))
        else:
            op=kron([ID]*i+[gate]+[ID]*(L-i-l))
        init=op@init
    return init
