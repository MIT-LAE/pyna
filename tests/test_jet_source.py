import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.jet_source import compute_jet_mixing_source_noise
from pyna.noise_tables import (
    FanNoiseTables, 
    CoreNoiseTables, 
    JetMixingNoiseTables
)

@pytest.mark.parametrize(
    "V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines, msap_jet_expected",
    [
        ()
    ]
)
def test_compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines, msap_jet_expected):

    msap_jet = compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines)
    assert_quantity_almost_equal(msap_jet, msap_jet_expected)

