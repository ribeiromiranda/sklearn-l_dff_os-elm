import numpy as np
from numpy.testing import assert_almost_equal

from ldffoselm import DffOsElm


def test_demo():
    X = np.random.random((100, 10))
    estimator = DffOsElm()
    estimator.fit(X, X[:, 0])
    assert_almost_equal(estimator.predict(X), X[:, 0]**2)