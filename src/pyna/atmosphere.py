import numpy as np

from pyna.constants import (
    _RATIO_SPECIFIC_HEATS_AIR,
    _GAS_CONSTANT_AIR,
    _LAPSE_RATE_TROPOSPHERE
)


def temperature_at_altitude(z, z_ref, T_ref, lapse=_LAPSE_RATE_TROPOSPHERE):
    """Compute temperature at altitude in the international standard atmosphere (ISA)
    given a reference altitude and temperature."""
    return T_ref + (z - z_ref) * lapse


def speed_of_sound(T):
    """Compute speed of sound.
    
    Parameters
    ----------
    T: float
        Ambient temperature.
    
    Returns
    -------
    float:
        Ambient speed of sound.
    """
    return np.sqrt(_RATIO_SPECIFIC_HEATS_AIR * _GAS_CONSTANT_AIR * T)
