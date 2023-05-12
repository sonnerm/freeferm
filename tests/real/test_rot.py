from freeferm.real import rot_sb_to_dense,rot_dense_to_sb
from freeferm.real import rot_sb_to_sparse,rot_sparse_to_sb
# from freeferm.real import rot_sb_to_circuit,rot_circuit_to_sb
import numpy.linalg as la
import numpy as np
import pytest

def test_rot_sb_dense_unitary(seed_rng):
    L=4
    ham_sb=np.random.random(size=(2*L,2*L))
    ham_sb=ham_sb+ham_sb.T
    u_sb=la.eigh(ham_sb)[1]#this is a orthogonal now
    u_dense=rot_sb_to_dense(u_sb)
    assert u_dense.shape[0]==2**L
    assert u_dense.T.conj()@u_dense==pytest.approx(np.eye(2**L))
    assert u_dense@u_dense.T.conj()==pytest.approx(np.eye(2**L))
    assert u_sb==pytest.approx(rot_dense_to_sb(u_dense))
    #Testing all 4 sectors of O(2n)
    u_dense=rot_sb_to_dense(-u_sb)
    assert -u_sb==pytest.approx(rot_dense_to_sb(u_dense))
    u_sb=u_sb@np.diag([-1]+[1]*(2*L-1))
    u_dense=rot_sb_to_dense(u_sb)
    assert u_sb==pytest.approx(rot_dense_to_sb(u_dense))
    u_dense=rot_sb_to_dense(-u_sb)
    assert -u_sb==pytest.approx(rot_dense_to_sb(u_dense))

def test_rot_sb_dense_special():
    L=2
    u_sb=np.array([[0.,0.,0.77065087,-0.63725759],
                   [0.,-1.,0.,0.],
                   [1.,0.,0.,0.],
                   [0.,0.,0.63725759,0.77065087]])
    u_dense=rot_sb_to_dense(u_sb)
    assert u_dense.shape[0]==2**L
    assert u_dense.T.conj()@u_dense==pytest.approx(np.eye(2**L))
    assert u_dense@u_dense.T.conj()==pytest.approx(np.eye(2**L))
    u_sbd=rot_dense_to_sb(u_dense)
    assert u_sb==pytest.approx(u_sbd,abs=1e-9) or u_sb==pytest.approx(-u_sbd,abs=1e-9)
    #Testing all 4 sectors of O(2n)
    u_dense=rot_sb_to_dense(-u_sb)
    u_sbd=rot_dense_to_sb(u_dense)
    assert -u_sb==pytest.approx(u_sbd,abs=1e-9) or -u_sb==pytest.approx(-u_sbd,abs=1e-9)
    u_sb=u_sb@np.diag([-1]+[1]*(2*L-1))
    u_dense=rot_sb_to_dense(u_sb)
    u_sbd=rot_dense_to_sb(u_dense)
    assert u_sb==pytest.approx(u_sbd,abs=1e-9) or u_sb==pytest.approx(-u_sbd,abs=1e-9)
    u_dense=rot_sb_to_dense(-u_sb)
    u_sbd=rot_dense_to_sb(u_dense)
    assert -u_sb==pytest.approx(u_sbd,abs=1e-9) or -u_sb==pytest.approx(u_sbd,abs=1e-9)

@pytest.mark.xfail(reason="Sparse not implemented yet")
def test_rot_sb_sparse_unitary(seed_rng):
    L=4
    ham_sb=1.0j*np.random.random(size=(2*L,2*L))
    ham_sb=ham_sb-ham_sb.T #antisymmetric
    u_sb=la.eigh(ham_sb)[1]#this is a unitary now
    u_sparse=rot_sb_to_sparse(u_sb)
    u_dense=rot_sb_to_dense(u_sb)
    assert u_sb==pytest.approx(rot_sparse_to_sb(u_sparse))
    assert u_sparse@np.eye(2**L)==pytest.approx(u_dense)
    #Testing all 4 sectors of O(2n)
    u_sparse=rot_sb_to_sparse(-u_sb)
    u_dense=rot_sb_to_dense(-u_sb)
    assert -u_sb==pytest.approx(rot_sparse_to_sb(u_sparse))
    assert u_sparse@np.eye(2**L)==pytest.approx(u_dense)
    u_sb=u_sb@np.diag([-1]+[1]*(2*L-1))
    u_sparse=rot_sb_to_sparse(u_sb)
    u_dense=rot_sb_to_dense(u_sb)
    assert u_sb==pytest.approx(rot_sparse_to_sb(u_sparse))
    assert u_sparse@np.eye(2**L)==pytest.approx(u_dense)
    u_sparse=rot_sb_to_sparse(-u_sb)
    u_dense=rot_sb_to_dense(-u_sb)
    assert -u_sb==pytest.approx(rot_sparse_to_sb(u_sparse))
    assert u_sparse@np.eye(2**L)==pytest.approx(u_dense)
