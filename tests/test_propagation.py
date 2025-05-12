import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal


from pyna.propagation import(
    compute_propagation,
    compute_ground_effects
)

# @pytest.mark.parametrize(
#     "msap_source, altitude, distance_source_observer, elevation_angle, density,  speed_of_sound_average, \
#     acoustic_impedance, frequency, n_frequency_subbands, observer_position, ground_resistance, incoherence_constant, \
#     flag_direct_propagation, flag_atmospheric_absorption, flag_ground_effects, msap_propagation_expected",
#     [
#         (),
#     ]
# )
# def test_compute_propagation(msap_source, altitude, distance_source_observer, elevation_angle, density,  speed_of_sound_average,
#     acoustic_impedance, frequency, n_frequency_subbands, observer_position, ground_resistance, incoherence_constant,
#     flag_direct_propagation, flag_atmospheric_absorption, flag_ground_effects):
#     pass

# @pytest.mark.parametrize(
#     "distance_source_observer, elevation_angle, observer_position, speed_of_sound_average, density, frequency, \
#     n_frequency_subbands, ground_resistance, incoherence_constant, ground_effects_expected",
#     [
#         (),
#     ]
# )
# def test_compute_ground_effects(distance_source_observer, elevation_angle, observer_position, speed_of_sound_average, density, frequency,
#     n_frequency_subbands, ground_resistance, incoherence_constant, ground_effects_expected):
#     pass
