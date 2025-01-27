import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.jet_source import compute_jet_mixing_source_noise
from pyna.noise_tables import (
    JetMixingNoiseTables,
    JetShockNoiseTables
)

# @pytest.mark.parametrize(
#     "V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines, msap_jet_expected",
#     [
#         ()
#     ]
# )
# def test_compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines, msap_jet_mixing_expected):

#     msap_jet_mixing = compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines)
#     assert_quantity_almost_equal(msap_jet_mixing, msap_jet_mixing_expected)


# @pytest.mark.parametrize(
#     "V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines, msap_jet_shock_expected",
#     [
#         ()
#     ]
# )
# def test_compute_jet_shock_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines, msap_jet_shock_expected):

#     msap_jet_shock = compute_jet_shock_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, A_e, p_ref, n_engines)
#     assert_quantity_almost_equal(msap_jet_shock, msap_jet_shock_expected)
