from freeferm.complex import quad_sb_to_dense,quad_dense_to_sb,quad_sb_to_sparse,quad_sparse_to_sb
import numpy.linalg as la
import numpy as np
import pytest

def test_quad_sb_dense_hermitian(seed_rng):
    L=4
    ham_sb=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    ham_sb=ham_sb.T.conj()+ham_sb
    ham_dense=quad_sb_to_dense(ham_sb)
    assert ham_sb==pytest.approx(quad_dense_to_sb(ham_dense))
    assert ham_dense.shape[0]==2**L
    assert ham_dense.T.conj()==pytest.approx(ham_dense)
    modes=la.eigvalsh(ham_sb)
    comev=la.eigvalsh(ham_dense)
    comev2=np.zeros((2**L))
    for i in range(2**L):
        comev2[i]=sum([modes[k] if s=="1" else 0 for k,s in enumerate(bin(i+2**L)[3:])])
    assert np.sort(comev)==pytest.approx(np.sort(comev2))

def test_quad_sb_sparse_hermitian(seed_rng):
    L=4
    ham_sb=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    ham_sb=ham_sb.T.conj()+ham_sb
    ham_sparse=quad_sb_to_sparse(ham_sb)
    assert ham_sb==pytest.approx(quad_sparse_to_sb(ham_sparse))
    ham_dense=quad_sb_to_dense(ham_sb)
    assert ham_sparse@np.eye(2**L)==pytest.approx(ham_dense)

def test_quad_sb_dense_generic(seed_rng):
    L=4
    ham_sb=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    assert ham_sb.T.conj()!=pytest.approx(ham_sb) #testcase is indeed generic
    ham_dense=quad_sb_to_dense(ham_sb)
    assert ham_sb==pytest.approx(quad_dense_to_sb(ham_dense))
    assert ham_dense.shape[0]==2**L
    assert ham_dense.T.conj()!=pytest.approx(ham_dense) #testcase is indeed generic
    ham_sb2=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    ham_dense2=quad_sb_to_dense(ham_sb2)
    alpha=np.random.random()+1.0j*np.random.random()
    assert ham_dense2+alpha*ham_dense==pytest.approx(quad_sb_to_dense(ham_sb2+alpha*ham_sb))

def test_quad_sb_sparse_generic(seed_rng):
    L=4
    ham_sb=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    ham_sparse=quad_sb_to_sparse(ham_sb)
    assert ham_sb==pytest.approx(quad_sparse_to_sb(ham_sparse))
    ham_dense=quad_sb_to_dense(ham_sb)
    assert ham_sparse@np.eye(2**L)==pytest.approx(ham_dense)
