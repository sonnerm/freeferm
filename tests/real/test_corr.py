import pytest
import numpy as np
import numpy.linalg as la
from freeferm import mps_to_dense,apply_circuit_to_dense,apply_circuit_to_mps,mps_vac,is_canonical
from freeferm.real import corr_vac,corr_full,dense_vac,dense_full
from freeferm.real import corr_to_dense,dense_to_corr,mps_to_corr,corr_to_mps,corr_to_circuit
from freeferm.real import quad_sb_to_dense

def test_corr_vac():
    L=4
    cv=corr_vac(L)
    assert cv==pytest.approx(dense_to_corr(dense_vac(L)))
    assert corr_to_dense(cv)==pytest.approx(dense_vac(L))
def test_corr_full():
    L=4
    cv=corr_full(L)
    assert cv==pytest.approx(dense_to_corr(dense_full(L)))
    assert corr_to_dense(cv)==pytest.approx(dense_full(L))
def test_corr_dense(seed_rng):
    L=6
    ham=1.0j*np.random.random(size=(2*L,2*L))
    ham=ham.T.conj()+ham
    ham=quad_sb_to_dense(ham)
    phi=la.eigh(ham)[1][:,0]
    corr=dense_to_corr(phi)
    correv=la.eigvalsh(corr)
    assert np.abs(correv)==pytest.approx(0.5)
    eve=corr_to_dense(corr)
    assert np.abs(eve.T.conj()@phi)==pytest.approx(1.0)

def test_corr_mps_short(seed_rng):
    L=2
    ham=1.0j*np.random.random(size=(2*L,2*L))
    ham=ham.T.conj()+ham
    ham=quad_sb_to_dense(ham)
    phi=la.eigh(ham)[1][:,0]
    corr=dense_to_corr(phi)
    correv=la.eigvalsh(corr)
    assert np.abs(correv)==pytest.approx(0.5)
    mpsi=corr_to_mps(corr)
    eve=mps_to_dense(mpsi)
    assert is_canonical(mpsi)
    assert np.abs(eve.T.conj()@phi)==pytest.approx(1.0)
def test_corr_mps_long(seed_rng):
    L=20
    corr=np.random.random(size=(2*L,2*L))
    corr=corr.T.conj()+corr
    rot=la.eigh(corr)[1]
    corr=rot.T@corr_vac(L)@rot
    mps=corr_to_mps(corr,chi=128)
    print([m.shape for m in mps])
    assert mps_to_corr(mps)==pytest.approx(corr)
