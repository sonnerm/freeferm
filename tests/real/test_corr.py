import pytest
from freeferm.real import corr_vac,corr_full,dense_vac,dense_full
from freeferm.real import corr_to_dense,dense_to_corr,mps_to_corr,corr_to_mps

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
