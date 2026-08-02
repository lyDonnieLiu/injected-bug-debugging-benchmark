import numpy as np

from common.seeding import set_seed


def test_numpy_seed_is_reproducible():
    set_seed(7)
    first = np.random.rand(5)
    set_seed(7)
    second = np.random.rand(5)
    assert np.array_equal(first, second)