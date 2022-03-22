import pytest
import numpy.linalg as la
from freeferm import mps_vac,mps_full,dense_vac,dense_full
from freeferm import apply_circuit_to_mps,apply_circuit_to_dense
from ttarray.raw import ttslice_to_dense,dense_to_ttslice
import numpy as np
def test_mps_vac():
    assert ttslice_to_dense(mps_vac(1))[0,:,0]==pytest.approx(dense_vac(1))
    assert ttslice_to_dense(mps_vac(2))[0,:,0]==pytest.approx(dense_vac(2))
    assert ttslice_to_dense(mps_vac(3))[0,:,0]==pytest.approx(dense_vac(3))
    assert ttslice_to_dense(mps_vac(4))[0,:,0]==pytest.approx(dense_vac(4))

def test_mps_full():
    assert ttslice_to_dense(mps_full(1))[0,:,0]==pytest.approx(dense_full(1))
    assert ttslice_to_dense(mps_full(2))[0,:,0]==pytest.approx(dense_full(2))
    assert ttslice_to_dense(mps_full(3))[0,:,0]==pytest.approx(dense_full(3))
    assert ttslice_to_dense(mps_full(4))[0,:,0]==pytest.approx(dense_full(4))
def test_apply_circuit(seed_rng):
    L=6
    vec=np.random.random(size=(2**L))+1.0j*np.random.random(size=(2**L))
    mps=dense_to_ttslice(vec[None,:,None],cluster=((2,),)*L)
    for s in range(1,L+1):
        for i in range(L-s+1):
            gate=np.random.random(size=(2**s,2**s))+np.random.random(size=(2**s,2**s))*1.0j
            gate=la.eigh(gate)[1]
            mps=apply_circuit_to_mps(mps,[(i,gate,False)])
            vec=apply_circuit_to_dense(vec,[(i,gate,False)])
            assert ttslice_to_dense(mps)==pytest.approx(vec[None,:,None])

    for s in range(1,L+1):
        for i in range(L-s+1):
            gate=np.random.random(size=(2**s,2**s))+np.random.random(size=(2**s,2**s))*1.0j
            gate=la.eigh(gate)[1]
            mps=apply_circuit_to_mps(mps,[(i,gate,True)])
            vec=apply_circuit_to_dense(vec,[(i,gate,True)])
            assert ttslice_to_dense(mps)==pytest.approx(vec[None,:,None])

def test_apply_circuit_truncate(seed_rng):
    L=10
    vec=np.random.random(size=(2**L))+1.0j*np.random.random(size=(2**L))
    mps=dense_to_ttslice(vec[None,:,None],cluster=((2,),)*L)
    for _ in range(20):
        for i in range(L-1):
            gate=np.random.random(size=(4,4))+np.random.random(size=(4,4))*1.0j
            gate=la.eigh(gate)[1]
            mps=apply_circuit_to_mps(mps,[(i,gate,False)],chi=200,cutoff=0.0)
            vec=apply_circuit_to_dense(vec,[(i,gate,False)])
            assert ttslice_to_dense(mps)==pytest.approx(vec[None,:,None])
            assert max(m.shape[0] for m in mps)<=200
