import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.levels import compute_spl

@pytest.mark.parametrize(
    "msap, rho_0, c_0, spl_expected",
    [
        (np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]),
         1.225, 
         340.,
         np.array([53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846, 53.02187846,
                   53.02187846, 53.02187846, 53.02187846, 53.02187846])),
        (np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20]),
         1.225,
         340., 
         -96.978122*np.ones(24))
    ]
)
def test_compute_spl(msap, rho_0, c_0, spl_expected):

    spl = compute_spl(msap, rho_0, c_0)
    assert_quantity_almost_equal(spl, spl_expected, atol=1e-20)