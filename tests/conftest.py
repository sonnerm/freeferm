import pytest
import numpy as np
import hashlib

@pytest.fixture(scope="function")
def seed_rng(request):
    '''
        Seeds the random number generator of numpy to a predictable value which
        is different for each test case
    '''
    stri=request.node.name+"_freeferm_test"
    np.random.seed(int.from_bytes(hashlib.md5(stri.encode('utf-8')).digest(),"big")%2**32)
    return None
