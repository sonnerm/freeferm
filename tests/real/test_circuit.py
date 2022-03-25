import pytest
import numpy as np
import numpy.linalg as la
from freeferm.real import rot_sb_to_circuit,rot_circuit_to_sb
from freeferm.real import corr_to_circuit,corr_vac,quad_sb_to_dense,dense_to_corr
from freeferm.real import apply_circuit_to_corr
def test_corr_to_circuit_short(seed_rng):
    L=6
    ham=1.0j*np.random.random(size=(2*L,2*L))
    ham=ham.T.conj()+ham
    ham=quad_sb_to_dense(ham)
    phi=la.eigh(ham)[1][:,0]
    corr=dense_to_corr(phi)
    circ=corr_to_circuit(corr)
    circ_bw=[(c[0],c[1].T.conj(),c[2],c[3].T) for c in circ[::-1]]
    assert apply_circuit_to_corr(corr,circ_bw)==pytest.approx(corr_vac(L))
    # print(apply_circuit_to_corr(corr_vac(L),circ))
    # print(corr)
    # print(circ)
    assert apply_circuit_to_corr(corr_vac(L),circ) == pytest.approx(corr)


def test_corr_to_circuit_long(seed_rng):
    L=20
    corr=np.random.random(size=(2*L,2*L))
    corr=corr.T.conj()+corr
    rot=la.eigh(corr)[1]
    corr=rot.T@corr_vac(L)@rot
    circ=corr_to_circuit(corr)
    circ_bw=[(c[0],c[1].T.conj(),c[2],c[3].T) for c in circ[::-1]]
    assert apply_circuit_to_corr(corr,circ_bw) == pytest.approx(corr_vac(L))
    assert apply_circuit_to_corr(corr_vac(L),circ) == pytest.approx(corr)

# def test_rot_sb_to_circuit(seed_rng):
#     raise NotImplementedError()
