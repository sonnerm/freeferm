import pytest
from freeferm import mps_to_dense,mps_vac,dense_vac,dense_to_mps,mpo_to_dense,dense_to_mpo
import numpy as np
def test_mps_vac():
    assert mps_to_dense(mps_vac(1))==pytest.approx(dense_vac(1))
    assert mps_to_dense(mps_vac(2))==pytest.approx(dense_vac(2))
    assert mps_to_dense(mps_vac(3))==pytest.approx(dense_vac(3))
    assert mps_to_dense(mps_vac(4))==pytest.approx(dense_vac(4))
def test_mps_dense(seed_rng):
    for L in range(1,4):
        vec=np.random.random(size=2**L)+1.0j*np.random.random(size=2**L)
        mps=dense_to_mps(vec)
        for i,m in enumerate(mps):
            assert m.shape[0]==2**(min(i,L-i))
            assert m.shape[1]==2
            assert m.shape[2]==2**(min(i+1,L-i-1))
            ms=m.reshape((m.shape[0]*m.shape[1],m.shape[2]))
            assert ms.T.conj()@ms == np.eye(ms.shape[1])
        assert mps_to_dense(mps)==pytest.approx(vec)

def test_mpo_dense(seed_rng):
    for L in range(1,4):
        mat=np.random.random(size=(2**L,2**L))+1.0j*np.random.random(size=(2**L,2**L))
        mpo=dense_to_mpo(mat)
        for i,m in enumerate(mpo):
            assert m.shape[0]==2**(min(i,L-i))
            assert m.shape[1]==2
            assert m.shape[2]==2
            assert m.shape[3]==2**(min(i+1,L-i-1))
            ms=m.reshape((m.shape[0]*m.shape[1]*m.shape[2],m.shape[3]))
            assert ms.T.conj()@ms == np.eye(ms.shape[1])
        assert mpo_to_dense(mpo)==pytest.approx(mat)
