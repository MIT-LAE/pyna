import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.noise_tables import (
    FanNoiseTables, 
    CoreNoiseTables, 
    JetMixingNoiseTables,
    JetShockNoiseTables
)

# Fan noise tables
@pytest.mark.parametrize(
    "theta, method, directivity_expected",
    [
        (0., "kresja", -0.5),
        (180., "kresja", -73)
    ]        
)
def test_fan_noise_tables_get_inlet_broadband_directivity(theta, method, directivity_expected):

    tables = FanNoiseTables()
    directivity = tables.get_inlet_broadband_directivity(theta, method)
    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "theta, method, directivity_expected",
    [
        (0., "alliedsignal", 0),
        (180., "alliedsignal", -20)
    ]        
)
def test_fan_noise_tables_get_discharge_broadband_directivity(theta, method, directivity_expected):
    
    tables = FanNoiseTables()
    directivity = tables.get_discharge_broadband_directivity(theta, method)
    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "theta, method, directivity_expected",
    [
        (0., "alliedsignal", -3),
        (180., "alliedsignal", -40.5)
    ]        
)
def test_fan_noise_tables_get_inlet_tones_directivity(theta, method, directivity_expected):
    
    tables = FanNoiseTables()
    directivity = tables.get_inlet_tones_directivity(theta, method)
    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "theta, method, directivity_expected",
    [
        (0., "alliedsignal", -34),
        (180., "alliedsignal", -16)
    ]        
)
def test_fan_noise_tables_get_discharge_tones_directivity(theta, method, directivity_expected):
    
    tables = FanNoiseTables()
    directivity = tables.get_discharge_tones_directivity(theta, method)
    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "theta, method, directivity_expected",
    [
        (0., "original", "-9.5"),
        (180., "original", -13.5)
    ]        
)
def test_fan_noise_tables_get_combination_tones_directivity(theta, method, directivity_expected):
    
    tables = FanNoiseTables()
    directivity = tables.get_combination_tones_directivity(theta, method)
    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "M_tip, subharmonic, method, tipmach_factor_expected",
    [
        (1.67, 1, "original", 68.38880870165691),
        (1.67, 2, "original", 72.42081345435197),
        (1.67, 3, "original", 70.40867525444706),
        (1., 1, "original", 30.0),
        (1., 2, "original", 30.0),
        (1., 3, "original", 30.0),
    ]        
)
def test_fan_noise_tables_get_combination_tones_tipmach(M_tip, subharmonic, method, tipmach_factor_expected):
    
    tables = FanNoiseTables()
    tipmach_factor = tables.get_combination_tones_tipmach(M_tip, subharmonic, method)
    
    assert_quantity_almost_equal(tipmach_factor, tipmach_factor_expected)

@pytest.mark.parametrize(
    "f_bpf, subharmonic, method, spectral_distribution_expected",
    [
        (0.1, 1, "original", -755.68),
        (0.1, 2, "original", -361.81),
        (0.1, 3, "original", -169.2),
        (1., 1, "original", 30.),
        (1., 2, "original", 30.),
        (1., 3, "original", 30.),
    ]        
)
def test_fan_noise_tables_get_combination_tones_spectral_distribution(f_bpf, subharmonic, method, spectral_distribution_expected):
    
    tables = FanNoiseTables()
    spectral_distribution = tables.get_combination_tones_tipmach(f_bpf, subharmonic, method)
    assert_quantity_almost_equal(spectral_distribution, spectral_distribution_expected)


# Core noise tables
@pytest.mark.parametrize(
    "theta, directivity_expected",
    [
        (0,    -0.85),
        (90.,  -0.16),
        (180., -0.90)
    ]
)
def test_core_noise_tables_get_directivity(theta, directivity_expected):

    tables = CoreNoiseTables()
    directivity = tables.get_directivity(theta)
    assert_quantity_almost_equal(directivity, directivity_expected)

@pytest.mark.parametrize(
    "log10_f_fp, spectral_distribution_expected",
    [
        (-1.1, -3.87),
        (0.0, -0.72),
        (1.6, -6.2)
    ]
)
def test_core_noise_tables_get_spectral_distribution(log10_f_fp, spectral_distribution_expected):

    tables = CoreNoiseTables()
    spectral_distribution = tables.get_spectral_distribution(log10_f_fp)
    assert_quantity_almost_equal(spectral_distribution, spectral_distribution_expected)


# Jet mixing noise tables
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


# Jet shock noise tables
@pytest.mark.parametrize(
    "log10_sigma, correlation_coefficient_spectrum_expected",
    [
        (-0.7, 0.703),
        (0., 0.735),
        (2.0, 0.015)
    ]
)
def test_jet_shock_noise_tables_get_correlation_coefficient_spectrum(log10_sigma, correlation_coefficient_spectrum_expected):

    tables = JetShockNoiseTables()
    correlation_coefficient_spectrum = tables.get_correlation_coefficient_spectrum(log10_sigma)
    assert_quantity_almost_equal(correlation_coefficient_spectrum, correlation_coefficient_spectrum_expected)

@pytest.mark.parametrize(
    "log10_sigma, group_source_strength_expected",
    [
        (0.0, -2.69),
        (1.0, -1.09),
        (2.5, -2.90)
    ]
)
def test_jet_shock_noise_tables_get_group_source_strength(log10_sigma, group_source_strength_expected):

    tables = JetShockNoiseTables()
    group_source_strength = tables.get_group_source_strength_spectrum(log10_sigma)
    assert_quantity_almost_equal(group_source_strength, group_source_strength_expected)