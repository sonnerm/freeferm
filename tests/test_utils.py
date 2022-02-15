import freeferm
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
    assert freeferm.check_sparse_lmax(freeferm.SPARSE_LMAX)

def test_basic_ops():
    pass
def test_kron():
    pass
def test_outer():
    pass
