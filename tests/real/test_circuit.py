import pytest
import numpy as np
import numpy.linalg as la
from freeferm.real import rot_sb_to_circuit,rot_circuit_to_sb
from freeferm.real import corr_to_circuit,mp_corr_to_circuit,flamp_corr_to_circuit,corr_vac,quad_sb_to_dense,dense_to_corr
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
def test_corr_to_circuit_long_mp(seed_rng):
    L=20
    corr=np.random.random(size=(2*L,2*L))
    corr=corr.T.conj()+corr
    rot=la.eigh(corr)[1]
    corr=rot.T@corr_vac(L)@rot
    circ=mp_corr_to_circuit(corr)
    circ_bw=[(c[0],c[1].T.conj(),c[2],c[3].T) for c in circ[::-1]]
    assert apply_circuit_to_corr(corr,circ_bw) == pytest.approx(corr_vac(L))
    assert apply_circuit_to_corr(corr_vac(L),circ) == pytest.approx(corr)
def test_corr_to_circuit_long_flamp(seed_rng):
    L=20
    corr=np.random.random(size=(2*L,2*L))
    corr=corr.T.conj()+corr
    rot=la.eigh(corr)[1]
    corr=rot.T@corr_vac(L)@rot
    circ=flamp_corr_to_circuit(corr)
    circ_bw=[(c[0],c[1].T.conj(),c[2],c[3].T) for c in circ[::-1]]
    assert apply_circuit_to_corr(corr,circ_bw) == pytest.approx(corr_vac(L))
    assert apply_circuit_to_corr(corr_vac(L),circ) == pytest.approx(corr)
# def test_corr_to_circuit_unstable(seed_rng):
#     import h5py
#     jcorr=np.array(h5py.File("/home/user/Exploration/impurity/data/Jx=0.3_Jy=0.3_g=0.0mu=0.0_del_t=1.0_L=200_IT_correlations.hdf5","r")["corr_t=30"])
#     L=jcorr.shape[0]//2
#     jcorrn=np.zeros_like(jcorr)
#     jcorrn[::2,::2]=jcorr[:jcorr.shape[0]//2,:jcorr.shape[1]//2]
#     jcorrn[1::2,::2]=jcorr[jcorr.shape[0]//2:,:jcorr.shape[1]//2]
#     jcorrn[1::2,1::2]=jcorr[jcorr.shape[0]//2:,jcorr.shape[1]//2:]
#     jcorrn[::2,1::2]=jcorr[:jcorr.shape[0]//2,jcorr.shape[1]//2:]
#     roti=np.diag([1,0]*L,1)[:-1,:-1]+np.diag([1j,0]*L,-1)[:-1,:-1]+np.diag([1,-1j]*L)
#     jcorrf=(roti@jcorrn@roti.T.conj())
#     jcorrf=(jcorrf-np.diag(np.diag(jcorrf)))/2
#     jcorrf[2::4,:]=-jcorrf[2::4,:]
#     jcorrf[3::4,:]=-jcorrf[3::4,:]
#     jcorrf[:,2::4]=-jcorrf[:,2::4]
#     jcorrf[:,3::4]=-jcorrf[:,3::4]
#     circ=corr_to_circuit(jcorrf,nbcutoff=1e-14)
#     circ_bw=[(c[0],c[1].T.conj(),c[2],c[3].T) for c in circ[::-1]]
#     print(la.eigvalsh(apply_circuit_to_corr(corr_vac(L),circ)))
#     print(np.max(np.abs(apply_circuit_to_corr(corr_vac(L),circ)-jcorrf)))
#     # print("circuit depth: %i"%len(circ))
#     assert apply_circuit_to_corr(jcorrf,circ_bw) == pytest.approx(corr_vac(L),abs=1e-7,rel=1e-7)
#     assert apply_circuit_to_corr(corr_vac(L),circ) == pytest.approx(jcorrf,abs=1e-7,rel=1e-7)

# def test_rot_sb_to_circuit(seed_rng):
#     raise NotImplementedError()
