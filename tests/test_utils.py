import freeferm
import numpy as np
import numpy.linalg as la
from freeferm import check_dense_lmax,check_sparse_lmax,SX,SY,SZ,Sp,Sm,ID,outer,kron,dense_vac,dense_full,eigu
import pytest
def test_check_lmax():
    assert freeferm.DENSE_LMAX<20
    with pytest.raises(ValueError):
        check_dense_lmax(20)
    assert freeferm.DENSE_LMAX>5
    check_dense_lmax(5)
    assert freeferm.SPARSE_LMAX>=freeferm.DENSE_LMAX
    assert freeferm.SPARSE_LMAX<40
    with pytest.raises(ValueError):
        check_sparse_lmax(40)
    check_sparse_lmax(freeferm.SPARSE_LMAX)

def test_basic_ops():
    #Tests commutation relations between the basic operators SX,SY,SZ,Sp,Sm,ID
    assert (SX@SY-SY@SX == 2.0j*SZ).all()
    assert (SZ@SX-SX@SZ == 2.0j*SY).all()
    assert (SY@SZ-SZ@SY == 2.0j*SX).all()
    assert (SZ@Sp-Sp@SZ == 2*Sp).all()
    assert (SZ@Sm-Sm@SZ == -2*Sm).all()
    assert (Sp@Sm-Sm@Sp == SZ).all()
    assert np.trace(SX) == 0.0
    assert np.trace(SY) == 0.0
    assert np.trace(SZ) == 0.0
    assert (SX == SX.T.conj()).all()
    assert (SY == SY.T.conj()).all()
    assert (SZ == SZ.T.conj()).all()
    assert (SX@SX == ID).all()
    assert (SY@SY == ID).all()
    assert (SZ@SZ == ID).all()
    assert (Sp==Sm.T.conj()).all()
def test_outer_kron():
    #Tests if outer and kron are consistent with the ordering
    sdo=np.array([0,1])
    smi=np.array([1,-1])/np.sqrt(2)
    state=outer([sdo,smi,sdo,smi])
    assert state@kron([SZ,SX,SZ,ID])@state==pytest.approx(-1.0)
    assert state@kron([ID,SZ,SX,SZ])@state==pytest.approx(0.0)
    assert state@kron([ID,SX,SZ,SX])@state==pytest.approx(-1.0)
    assert state@kron([SZ,SX,SZ,SX])@state==pytest.approx(1.0)
    assert state@kron([SX,SZ,SX,SZ])@state==pytest.approx(0.0)
def test_dense_vac():
    #Tests if vacuum state works as expected
    assert len(dense_vac(1))==2 #size should be 2**L
    assert len(dense_vac(2))==4
    assert len(dense_vac(3))==8
    assert (outer([dense_vac(1),dense_vac(2),dense_vac(2)])==dense_vac(5)).all()#correct outer product
    assert (dense_vac(2)@dense_vac(2) == 1.0).all() #Normalized
    assert (dense_vac(1)@SZ@dense_vac(1)==-1.0).all() #Consistent with SZ
    assert (dense_vac(2)@kron([SZ,ID])@dense_vac(2)==-1.0).all() #Consistent with SZ
    assert (dense_vac(2)@kron([ID,SZ])@dense_vac(2)==-1.0).all()
    assert (dense_vac(2)@kron([SZ,SZ])@dense_vac(2)==1.0).all()
    assert (Sm@dense_vac(1) == np.array([0.0,0.0])).all()
    assert (Sp@dense_vac(1) == dense_full(1)).all()

def test_dense_full():
    #Tests if full state works as expected
    assert len(dense_full(1))==2 #size should be 2**L
    assert len(dense_full(2))==4
    assert len(dense_full(3))==8
    assert (outer([dense_full(1),dense_full(2),dense_full(2)])==dense_full(5)).all()#correct outer product
    assert (dense_full(2)@dense_full(2) == 1.0).all() #Normalized
    assert (dense_full(1)@SZ@dense_full(1)==1.0).all() #Consistent with SZ
    assert (dense_full(2)@kron([SZ,ID])@dense_full(2)==1.0).all() #Consistent with SZ
    assert (dense_full(2)@kron([ID,SZ])@dense_full(2)==1.0).all()
    assert (dense_full(2)@kron([SZ,SZ])@dense_full(2)==1.0).all()
    assert (Sp@dense_full(1) == np.array([0.0,0.0])).all()
    assert (Sm@dense_full(1) == dense_vac(1)).all()
    assert (dense_vac(3)@dense_full(3) == 0.0).all()
def test_eigu(seed_rng):
    L=20
    herm=np.random.random(size=(L,L))+1.0j*np.random.random(size=(L,L))
    herm=herm+herm.T.conj()
    _,U=la.eigh(herm)
    eveu,evveu=eigu(U)
    ev,evv=la.eig(U)
    assert np.abs(ev)==pytest.approx(1.0)
    assert np.abs(eveu)==pytest.approx(1.0)
    assert ev[np.argsort(np.angle(ev))]==pytest.approx(eveu[np.argsort(np.angle(eveu))])
    M=(evv[:,np.argsort(np.angle(ev))].T.conj()@evveu[:,np.argsort(np.angle(eveu))])
    assert M==pytest.approx(np.diag(np.diag(M)))
