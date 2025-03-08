import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.noise_tables import (
    FanNoiseTables, 
    CoreNoiseTables, 
    JetMixingNoiseTables,
    JetShockNoiseTables,
    AirframeNoiseTables,
    PropagationTables
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

@pytest.mark.parametrize(
    "method, flight_segment, i_harmonic, theta, cleanup_tcs_expected",
    [
        ("geae", "takeoff", 1, 90, 2.6),
        ("geae", "takeoff", 1, 10, 4.8),
        ("geae", "takeoff", 1, 170, 3.5),
        ("geae", "takeoff", 2, 90, 1.1),
        ("geae", "takeoff", 2, 10, 5.8),
        ("geae", "takeoff", 2, 170, 0.8),
        ("geae", "takeoff", 3, 90, 0),
        ("geae", "takeoff", 3, 10, 0),
        ("geae", "takeoff", 3, 170, 0),
        ("original", "takeoff", 1, 90, 0),
        ("original", "takeoff", 1, 10, 0),
        ("original", "takeoff", 1, 170, 0),
    ]
)
def test_get_cleanup_turbulent_control_structures(method, flight_segment, i_harmonic, theta, cleanup_tcs_expected):

    tables = FanNoiseTables()
    cleanup_tcs = tables.get_cleanup_turbulent_control_structures(method, flight_segment, i_harmonic, theta)
    assert_quantity_almost_equal(cleanup_tcs, cleanup_tcs_expected)

@pytest.mark.parametrize(
    "frequency, theta, noise_direction, suppression_expected",
    [
        (
            [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
            90, 
            "inlet",
            [0.852196, 0.808852, 0.756521, 0.701186, 0.64056, 0.569043, 0.502395, 0.436044, 0.369959, 0.306775, 0.254126, 0.207146, 0.167071, 0.137239, 0.11418, 0.095671, 0.084475, 0.077956, 0.0758635, 0.0788859, 0.0872591, 0.103179, 0.130114, 0.168291]
        ),
        (
            [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
            90, 
            "discharge",
            [0.852196, 0.808852, 0.756521, 0.701186, 0.64056, 0.569043, 0.502395, 0.436044, 0.369959, 0.306775, 0.254126, 0.207146, 0.167071, 0.137239, 0.11418, 0.095671, 0.084475, 0.077956, 0.0758635, 0.0788859, 0.0872591, 0.103179, 0.130114, 0.168291]
        )
    ]
)
def test_get_liner_suppression(frequency, theta, noise_direction, suppression_expected):

    tables = FanNoiseTables()
    suppression = tables.get_liner_suppression(frequency, theta, noise_direction)
    assert_quantity_almost_equal(suppression, suppression_expected)

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

@pytest.mark.parametrize(
    "frequency, theta, suppression_expected",
    [
        (
            [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
            90, 
            [0.063096, 0.079433, 0.1, 0.12589, 0.15849, 0.15849, 0.12589, 0.1122, 0.12589, 0.12589, 0.12589, 0.1, 0.1, 0.1, 0.12589, 0.12589, 0.12589, 0.12589, 0.12589, 0.12589, 0.12589, 0.12589, 0.12589, 0.15849]
        ),
        (
            [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
            10, 
            [0.020893, 0.02884, 0.039811, 0.054954, 0.075858, 0.075858, 0.054954, 0.046774, 0.054954, 0.054954, 0.054954, 0.039811, 0.039811, 0.039811, 0.054954, 0.054954, 0.054954, 0.054954, 0.054954, 0.054954, 0.054954, 0.054954, 0.054954, 0.075858]
        ),
        (
            [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
            170, 
            [15.849, 12.589, 10, 7.9433, 6.3096, 6.3096, 7.9433, 8.9125, 7.9433, 7.9433, 7.9433, 10, 10, 10, 7.9433, 7.9433, 7.9433, 7.9433, 7.9433, 7.9433, 7.9433, 7.9433, 7.9433, 6.3096]
        )
    ]
)
def test_get_high_speed_research_suppression(frequency, theta, suppression_expected):

    tables = AirframeNoiseTables()
    suppression = tables.get_high_speed_research_suppression(frequency, theta)
    assert_quantity_almost_equal(suppression, suppression_expected)


@pytest.mark.parametrize(
    "altitude, frequency, absorption_expected",
    [
        (0., 50., 0.000285538),
        (6000., 10000., 0.08914042),
    ]
)
def test_get_atmospheric_absorption(altitude, frequency, absorption_expected):

    tables = PropagationTables()
    absorption = tables.get_atmospheric_absorption(altitude, frequency)
    assert_quantity_almost_equal(absorption, absorption_expected)