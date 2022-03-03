import pytest
import numpy.linalg as la
from freeferm import mps_to_dense,dense_to_mps,is_canonical,compress_svd
from freeferm import mpo_to_dense,dense_to_mpo,kron
from freeferm import mps_slice_to_dense,dense_to_mps_slice
from freeferm import mpo_slice_to_dense,dense_to_mpo_slice
from freeferm import mps_vac,mps_full,dense_vac,dense_full
from freeferm import apply_circuit_to_mps,apply_circuit_to_dense
import numpy as np
def test_mps_vac():
    assert mps_to_dense(mps_vac(1))==pytest.approx(dense_vac(1))
    assert mps_to_dense(mps_vac(2))==pytest.approx(dense_vac(2))
    assert mps_to_dense(mps_vac(3))==pytest.approx(dense_vac(3))
    assert mps_to_dense(mps_vac(4))==pytest.approx(dense_vac(4))

def test_mps_full():
    assert mps_to_dense(mps_full(1))==pytest.approx(dense_full(1))
    assert mps_to_dense(mps_full(2))==pytest.approx(dense_full(2))
    assert mps_to_dense(mps_full(3))==pytest.approx(dense_full(3))
    assert mps_to_dense(mps_full(4))==pytest.approx(dense_full(4))
def test_mps_dense(seed_rng):
    for L in range(1,4):
        vec=np.random.random(size=2**L)+1.0j*np.random.random(size=2**L)
        mps=dense_to_mps(vec)
        for i,m in enumerate(mps):
            assert m.shape[0]==2**(min(i,L-i))
            assert m.shape[1]==2
            assert m.shape[2]==2**(min(i+1,L-i-1))
            if i!=L-1: #Last matrix contains norm
                ms=m.reshape((m.shape[0]*m.shape[1],m.shape[2]))
                assert ms.T.conj()@ms == pytest.approx(np.eye(ms.shape[1]))
        assert is_canonical(mps)
        assert mps_to_dense(mps)==pytest.approx(vec)
        assert mps_to_dense(compress_svd(mps,chi=1024)) == pytest.approx(vec)
def test_mps_dense_slice(seed_rng):
    for d in [2,3]:
        for L in range(1,4):
            vec=np.random.random(size=(3,d**L,2))+1.0j*np.random.random(size=(3,d**L,2))
            mps=dense_to_mps_slice(vec,d)
            for i,m in enumerate(mps):
                assert m.shape[0]==min(d**i*3,d**(L-i)*2)
                assert m.shape[1]==d
                assert m.shape[2]==min(d**(i+1)*3,d**(L-i-1)*2)
                if i!=L-1: #Last matrix contains norm
                    ms=m.reshape((m.shape[0]*m.shape[1],m.shape[2]))
                    assert ms.T.conj()@ms == pytest.approx(np.eye(ms.shape[1]))
            assert is_canonical(mps)
            assert mps_slice_to_dense(mps)==pytest.approx(vec)
            assert mps_slice_to_dense(compress_svd(mps,chi=1024)) == pytest.approx(vec)
            if d==2:
                mps[-1]=np.einsum("adc,bd->abc",mps[-1],np.diag([1,-1]))
                vecm=np.einsum("abc,bd->abc",vec,np.kron(np.eye(2**(L-1)),np.diag([1,-1])))
                # vecm=vec
                assert mps_slice_to_dense(mps)==pytest.approx(vecm)


def test_mpo_dense(seed_rng):
    for L in range(1,4):
        mat=np.random.random(size=(2**L,2**L))+1.0j*np.random.random(size=(2**L,2**L))
        mpo=dense_to_mpo(mat)
        for i,m in enumerate(mpo):
            assert m.shape[0]==4**(min(i,L-i))
            assert m.shape[1]==2
            assert m.shape[2]==2
            assert m.shape[3]==4**(min(i+1,L-i-1))
            if i!=L-1:
                ms=m.reshape((m.shape[0]*m.shape[1]*m.shape[2],m.shape[3]))
                assert ms.T.conj()@ms == pytest.approx(np.eye(ms.shape[1]))
        assert mpo_to_dense(mpo)==pytest.approx(mat)

def test_mpo_dense_slice(seed_rng):
    for di,do,L in [(2,2,1),(3,2,1),(3,3,1),(2,3,1),(2,2,2),(3,2,2),(2,2,3)]:
        vec=np.random.random(size=(3,do**L,di**L,2))+1.0j*np.random.random(size=(3,do**L,di**L,2))
        mpo=dense_to_mpo_slice(vec,do,di)
        for i,m in enumerate(mpo):
            assert m.shape[0]==min((do*di)**i*3,(di*do)**(L-i)*2)
            assert m.shape[1]==do
            assert m.shape[2]==di
            assert m.shape[3]==min((do*di)**(i+1)*3,(do*di)**(L-i-1)*2)
            if i!=L-1: #Last matrix contains norm
                ms=m.reshape((m.shape[0]*m.shape[1]*m.shape[2],m.shape[3]))
                assert ms.T.conj()@ms == pytest.approx(np.eye(ms.shape[1]))
        assert mpo_slice_to_dense(mpo)==pytest.approx(vec)
def test_apply_circuit(seed_rng):
    L=6
    vec=np.random.random(size=(2**L))+1.0j*np.random.random(size=(2**L))
    mps=dense_to_mps(vec)
    for s in range(1,L+1):
        for i in range(L-s+1):
            gate=np.random.random(size=(2**s,2**s))+np.random.random(size=(2**s,2**s))*1.0j
            gate=la.eigh(gate)[1]
            mps=apply_circuit_to_mps(mps,[(i,gate,False)])
            vec=apply_circuit_to_dense(vec,[(i,gate,False)])
            assert is_canonical(mps)
            assert mps_to_dense(mps)==pytest.approx(vec)

    for s in range(1,L+1):
        for i in range(L-s+1):
            gate=np.random.random(size=(2**s,2**s))+np.random.random(size=(2**s,2**s))*1.0j
            gate=la.eigh(gate)[1]
            mps=apply_circuit_to_mps(mps,[(i,gate,True)])
            vec=apply_circuit_to_dense(vec,[(i,gate,True)])
            assert is_canonical(mps)
            assert mps_to_dense(mps)==pytest.approx(vec)
