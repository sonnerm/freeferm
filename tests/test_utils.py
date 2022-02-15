import freeferm
import numpy as np
from freeferm import check_dense_lmax,check_sparse_lmax,SX,SY,SZ,Sp,Sm,ID,outer,kron,dense_vac
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
