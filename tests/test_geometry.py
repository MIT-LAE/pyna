import pytest
import numpy as np
from pint.testsuite.helpers import assert_quantity_almost_equal

from pyna.geometry import(
    compute_source_observer_distance,
    compute_elevation_angle,
    compute_directivy_angles,
    compute_average_speed_of_sound_source_observer,
    compute_observer_time
)

# Inputs
n_timestamps = 2
_X = np.linspace(0, 10000, n_timestamps)
_Y = np.linspace(0, 10000, n_timestamps)
_Z = np.linspace(0, 10000, n_timestamps)
_ALPHA = np.linspace(1, 10, n_timestamps) * np.pi/180
_GAMMA = np.linspace(5, 10, n_timestamps) * np.pi/180
_T_SOURCE = np.linspace(0, 100, n_timestamps)
_T0 = np.linspace(300., 300, n_timestamps)
_OBSERVER_POSITION = np.array([6500., 0, 1.2])
_SOURCE_Z_OFFSET = 4.

_R_OBSERVER_DISTANCE = np.linspace(1, 10000, n_timestamps)
_AVERAGE_SPEED_OF_SOUND = np.linspace(350, 340, n_timestamps)

@pytest.mark.parametrize(
    "x, y, z, observer_position, source_z_offset, r_observer_expected",
    [
        (_X, _Y, _Z, _OBSERVER_POSITION, _SOURCE_Z_OFFSET, np.array([ 6500.000603, 14570.724342]))
    ]
)
def test_compute_source_observer_distance(x, y, z, observer_position, source_z_offset, r_observer_expected):
    r_observer = compute_source_observer_distance(x, y, z, observer_position, source_z_offset)
    assert_quantity_almost_equal(r_observer, r_observer_expected)


@pytest.mark.parametrize(
    "z, r, elevation_angle_expected",
    [
        (_Z, _R_OBSERVER_DISTANCE, np.array([0., 1.57079633]))
    ]
)
def test_compute_elevation_angle(z, r, elevation_angle_expected):
    elevation_angle = compute_elevation_angle(z, r)
    assert_quantity_almost_equal(elevation_angle, elevation_angle_expected)
    

@pytest.mark.parametrize(
    "x, y, z, observer_position, alpha, gamma, source_z_offset, polar_directivity_expected, azimuthal_directivity_expected",
    [
        (_X, _Y, _Z, _OBSERVER_POSITION, _ALPHA, _GAMMA, _SOURCE_Z_OFFSET, np.array([0.10515052, 2.04937519]), np.array([0., -0.88383]))
    ]
)
def test_compute_directivity_angles(x, y, z, observer_position, alpha, gamma, source_z_offset, polar_directivity_expected, azimuthal_directivity_expected):
    polar_angle, azimuthal_angle = compute_directivy_angles(x, y, z, observer_position, alpha, gamma, source_z_offset)
    assert_quantity_almost_equal(polar_angle, polar_directivity_expected)
    assert_quantity_almost_equal(azimuthal_angle, azimuthal_directivity_expected)

@pytest.mark.parametrize(
    "z, T_0, z_observer, c_average_expected",
    [
        (_Z, _T0, _OBSERVER_POSITION[2], np.array([347.186453, 365.334204]))
    ]
)
def test_compute_average_speed_of_sound_source_observer(z, T_0, z_observer, c_average_expected):
    c_average = compute_average_speed_of_sound_source_observer(z, T_0, z_observer)
    assert_quantity_almost_equal(c_average, c_average_expected)

@pytest.mark.parametrize(
    "t_source, distance_source_observer, average_speed_of_sound, t_observer_expected",
    [
        (_T_SOURCE, _R_OBSERVER_DISTANCE, _AVERAGE_SPEED_OF_SOUND, np.array([0.002857142857142857, 129.41176470588235]))
    ]
)
def test_compute_observer_time(t_source, distance_source_observer, average_speed_of_sound, t_observer_expected):
    t_observer = compute_observer_time(t_source, distance_source_observer, average_speed_of_sound)
    assert_quantity_almost_equal(t_observer, t_observer_expected)