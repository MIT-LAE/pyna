import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.noise_tables import JetMixingNoiseTables


@pytest.mark.parametrize(
        "log10_V_j_star, density_exponent_expected",
        [
            (-0.45, -1),
            (0., 0.77),
            (0.25, 2.),
         ]
)
def test_jet_mixing_noise_tables_get_density_exponent(log10_V_j_star, density_exponent_expected):

    tables = JetMixingNoiseTables()

    density_exponent = tables.get_density_exponent(log10_V_j_star)

    assert_quantity_almost_equal(density_exponent, density_exponent_expected)

@pytest.mark.parametrize(
    "log10_V_j_star, power_deviation_factor_expected",
    [
        (-0.4, -0.13),
        (0., 0.),
        (0.4, 0.14)
    ]
)
def test_jet_mixing_noise_tables_get_power_deviation_factor(log10_V_j_star, power_deviation_factor_expected):

    tables = JetMixingNoiseTables()

    power_deviation_factor = tables.get_power_deviation_factor(log10_V_j_star)

    assert_quantity_almost_equal(power_deviation_factor, power_deviation_factor_expected)

@pytest.mark.parametrize(
    "log10_V_j_star, theta, directivity_expected",
    [
        (0, -0.4, -0.48),
        (180., -0.4, 0.45),
        (90., -0.2, -0.25),
        (0., 0., -0.86),
        (180., 0., 0.46)
    ]
)
def test_jet_mixing_noise_tables_get_directivity(log10_V_j_star, theta, directivity_expected):

    tables = JetMixingNoiseTables()

    directivity = tables.get_directivity(log10_V_j_star, theta)

    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "V_j_star, theta, strouhal_correction_expected",
    [
        (1.,  90., 1.),
        (1.4, 120., 1.) ,
        (2.5, 120., 1.),
        (2., 140., 0.9),
        (1.4, 170., 1.),
        (2.5, 170., 0.27),
    ]
)
def test_jet_mixing_noise_tables_get_strouhal_correction(V_j_star, theta, strouhal_correction_expected):

    tables = JetMixingNoiseTables()

    strouhal_correction = tables.get_strouhal_correction(V_j_star, theta)

    assert_quantity_almost_equal(strouhal_correction, strouhal_correction_expected)

@pytest.mark.parametrize(
    "theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands, spectral_distribution_expected",
    [
        (90.,  1.0, 0.1,  -2.0,   24, 24*[41.4]),
        (180., 2.0, 0.1,   0.0,   24, 24*[22.]),
        (160., 2.0, 0.125, 1.7,   24, 24*[53.167]),
        (90.,  2.5, 0.175, 0.477, 24, 24*[14.7]),
        (120., 3.0, 0.125, -2.0,  24, 24*[43.167])
    ]
)
def test_jet_mixing_noise_tables_get_spectral_distribution(theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands, spectral_distribution_expected):
    
    tables = JetMixingNoiseTables()

    spectral_distribution = tables.get_spectral_distribution(theta, Tt_j_star, log10_V_j_star, log10_St, n_frequency_bands)

    assert_quantity_almost_equal(spectral_distribution, spectral_distribution_expected)

@pytest.mark.parametrize(
    "theta, forward_velocity_index_expected",
    [
        (0.0, 3.0),
        (90.0, 1.0),
        (180., 8.5)
    ]
)
def test_jet_mixing_noise_tables_get_forward_velocity_index(theta, forward_velocity_index_expected):

    tables = JetMixingNoiseTables()

    forward_velocity_index = tables.get_forward_velocity_index(theta)

    assert_quantity_almost_equal(forward_velocity_index, forward_velocity_index_expected)