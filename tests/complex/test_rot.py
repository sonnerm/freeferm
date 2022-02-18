from freeferm.complex import rot_sb_to_dense,rot_dense_to_sb
from freeferm.complex import rot_sb_to_sparse,rot_sparse_to_sb
import numpy.linalg as la
import numpy as np
import pytest
@pytest.mark.xfail(reason="Not Implemented yet")
def test_rot_sb_dense_unitary(seed_rng):
    L=4
    ham_sb=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    ham_sb=ham_sb+ham_sb.T.conj()
    u_sb=la.eigh(ham_sb)[1]#this is a unitary now
    u_dense=rot_sb_to_dense(u_sb)
    assert u_sb==pytest.approx(rot_dense_to_sb(u_dense))
    assert u_dense.shape[0]==2**L
    assert u_dense.T.conj()@u_dense==pytest.approx(np.eye(2**L))
    assert u_dense@u_dense.T.conj()==pytest.approx(np.eye(2**L))
    #Testing all 4 sectors of O(2n)
    u_dense=rot_sb_to_dense(-u_sb)
    assert -u_sb==pytest.approx(rot_dense_to_sb(u_dense))
    u_sb=u_sb@np.diag([-1]+[1]*(2*L-1))
    u_dense=rot_sb_to_dense(u_sb)
    assert u_sb==pytest.approx(rot_dense_to_sb(u_dense))
    u_dense=rot_sb_to_dense(-u_sb)
    assert -u_sb==pytest.approx(rot_dense_to_sb(u_dense))
@pytest.mark.xfail(reason="Sparse not implemented yet")
def test_rot_sb_sparse_unitary(seed_rng):
    L=4
    ham_sb=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    ham_sb=ham_sb+ham_sb.T.conj()
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
