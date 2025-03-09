import numpy as np
from scipy import special

from pyna.constants import (
    _R_SOURCE,
    _ACOUSTIC_IMPEDANCE_SEALEVEL
)

from pyna.noise_tables import PropagationTables

from pyna.frequency_bands import (
    get_frequency_subbands,
    get_spectrum_subbands
)

from pyna.constants import (
    _R_SOURCE,
    _ACOUSTIC_IMPEDANCE_SEALEVEL
)


def compute_propagation(
        msap_source, 
        altitude, 
        distance_source_observer, 
        elevation_angle,
        density, 
        speed_of_sound_average, 
        acoustic_impedance, 
        frequency, 
        n_frequency_subbands, 
        observer_position,
        ground_resistance,
        incoherence_constant,
        flag_direct_propagation, 
        flag_atmospheric_absorption, 
        flag_ground_effects
    ):
    
    """
    Computes direct propagation of mean-square acoustic pressure (msap):

    * Direct propagation (distance-law: R^2)
    * Characteristic impedance law
    * Atmospheric absorption

    Computes atmospheric absorption and ground reflections of propagated mean-square acoustic pressure.

    Parameters
    ----------
    msap_source : np.ndarray

    altitude : float
    distance_source_observer : float
    elevation_angle : float
    density : float
    speed_of_sound_average : float
    acoustic_impedance : float
    frequency : float
    n_frequency_subbands : int
    observer_position : np.ndarray
    flag_direct_propagation : bool
    flag_atmospheric_absorption : bool
    flag_ground_effects : bool

    Returns
    -------
    msap_prop : np.ndarray
        
    """

    tables = PropagationTables()

    # Apply spherical spreading and characteristic impedance effects to the MSAP
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
    if flag_direct_propagation:
        msap_source *= (_R_SOURCE ** 2 / distance_source_observer ** 2) * (_ACOUSTIC_IMPEDANCE_SEALEVEL / acoustic_impedance)
    
    # Allocate
    msap_prop = np.zeros(msap_source.size)

    if flag_atmospheric_absorption or flag_ground_effects:
        msap_sb = get_spectrum_subbands(msap_source, n_frequency_subbands)

        if flag_atmospheric_absorption:
            alpha_f = tables.get_atmospheric_absorption(altitude, frequency)
            
            # Compute absorption (convert dB to Np: 1dB is 0.115Np)
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
            msap_sb *= np.exp(-2 * 0.115 * alpha_f * (distance_source_observer - _R_SOURCE))

        if flag_ground_effects:
            msap_sb *= compute_ground_effects(distance_source_observer, elevation_angle, observer_position, speed_of_sound_average, density, frequency, n_frequency_subbands, ground_resistance, incoherence_constant)
                
        # Compute absorbed msap by adding up the msap at all the sub-band frequencies
        # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 22
        for j in np.arange(msap_source.size):
            msap_prop[j] = np.sum(msap_sb[j*n_frequency_subbands:(j+1)*n_frequency_subbands])

    else:
        msap_prop = msap_source

    return msap_prop


def compute_ground_effects(distance_source_observer, elevation_angle, observer_position, speed_of_sound_average, density, frequency, n_frequency_subbands, ground_resistance, incoherence_constant):
    """
    Compute the ground reflection coefficients.

    Parameters
    ----------
    distance_source_observer : float
    elevation_angle : float
    observer_position : np.ndarray
    speed_of_sound_average : float
    density : float
    frequency : np.ndarray
    n_frequency_subbands : int
    ground_resistance : float
    incoherence_constant : float

    Returns
    -------
    np.ndarray : 
        Ground reflection coefficients

    """

    frequency_subbands = get_frequency_subbands(frequency, n_frequency_subbands)

    # Compute difference in direct and reflected distance between source and observer
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 16
    distance_source_observer_reflected = (
        np.sqrt(distance_source_observer ** 2 + 4 * observer_position[2] ** 2 + 4 * distance_source_observer * observer_position[2] * np.sin(elevation_angle * np.pi / 180.))
    )
    dr = distance_source_observer_reflected - distance_source_observer

    # Compute wave number
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
    k = 2 * np.pi * frequency_subbands / speed_of_sound_average

    # Compute dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
    eta = 2 * np.pi * density * frequency_subbands / ground_resistance

    # Compute the cosine of the incidence angle
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 18
    cos_theta = (distance_source_observer * np.sin(elevation_angle * np.pi / 180.) + 2 * observer_position[2]) / distance_source_observer_reflected

    # Complex specific ground admittance nu
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 13 / adapted through Berton lateral attenuation paper
    nu = (1 + (6.86 * eta) ** (-0.75) + (4.36 * eta) ** (-0.73) * 1j) ** (-1)

    # Compute Gamma
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 5
    Gamma = (cos_theta - nu) / (cos_theta + nu)

    # Compute tau
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 9
    tau = np.sqrt(k * distance_source_observer_reflected / 2j) * (cos_theta + nu)

    # Compute complex spherical wave reflection coefficient
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 12
    U = np.real(tau ** 0)
    U[np.where(-np.real(tau) == 0)] = 0.5 * np.ones(n_frequency_subbands * frequency.size, dtype=np.float64)[np.where(-np.real(tau) == 0)]
    U[np.where(-np.real(tau) < 0)] = np.zeros(n_frequency_subbands * frequency.size, dtype=np.float64)[np.where(-np.real(tau) < 0)]

    # Compute F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 11
    F = -2 * np.sqrt(np.pi) * U * tau * np.exp(tau ** 2) + 1. / (2. * tau ** 2) - 3. / (2 * tau ** 2) ** 2

    # Compute complex spherical wave function F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 10 (using Faddeeva function)
    F[np.where(np.absolute(tau)<10)] = (1 - np.sqrt(np.pi) * tau * (special.wofz(tau * 1j)))[np.where(np.absolute(tau)<10)]

    # Compute Z_cswfc
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 6
    Z_cswfc = Gamma + (1 - Gamma) * F
    R = np.absolute(Z_cswfc)
    alpha = np.angle(Z_cswfc)

    # Compute the constant K and constant epsilon
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 16-17
    K = 2 ** (1. / (6. * n_frequency_subbands))
    eps = K - 1

    # Compute G
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 18
    if dr > 0:
        return 1 + R ** 2 + 2 * R * np.exp(-(incoherence_constant * k * dr) ** 2) * np.cos(alpha + k * dr) * np.sin(eps * k * dr) / (eps * k * dr)
    else:
        return 1 + R ** 2 + 2 * R * np.exp(-(incoherence_constant * k * dr) ** 2) * np.cos(alpha + k * dr)