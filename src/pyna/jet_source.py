import numpy as np
from pyna.noise_tables import (
    JetMixingNoiseTables,
    JetShockNoiseTables
)

_R_SOURCE = 0.3048


def compute_jet_mixing_source_noise(V_j_star, rho_j_star, A_j_star, Tt_j_star, theta, M_0, c_0, f, area_jet_effective, p_ref, n_engines):
    """
    Compute jet mixing noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    V_j_star : float

    rho_j_star : float

    A_j_star : float

    Tt_j_star : float

    theta : float

    M_0 : float

    c_0 : float

    f : np.ndarray

    area_jet_effective : float

    p_ref : float

    n_engines: int

            
    Returns
    -------
    np.ndarray
        Mean-square acoustic pressure of jet mixing source noise

    """

    tables = JetMixingNoiseTables()

    r_s_star = _R_SOURCE / np.sqrt(area_jet_effective)
    jet_delta = 0.

    # Calculate density exponent (omega)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
    omega = tables.get_density_exponent(np.log10(V_j_star))

    # Calculate power deviation factor (P)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
    p = tables.get_power_deviation_factor(np.log10(V_j_star))

    # Calculate acoustic power (Pi_star)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
    K = 6.67e-5
    Pi_star = K * rho_j_star ** omega * V_j_star ** 8 * p

    # Calculate directivity function (D)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
    d_function = tables.get_directivity(theta, np.log10(V_j_star))

    # Calculate Strouhal frequency adjustment factor (xi)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
    # TODO: check xi = min(1, xi)
    xi = tables.get_strouhal_correction(V_j_star, theta)

    # Calculate Strouhal number (St)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Eq. 9
    D_j_star = np.sqrt(4 * A_j_star / np.pi)  # Jet diamater [-] (rel. to sqrt(area_jet_effective))
    f_star = f * np.sqrt(area_jet_effective) / c_0
    St = (f_star * D_j_star) / (xi * (V_j_star - M_0))

    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
    f_function = tables.get_spectral_distribution(theta, Tt_j_star, np.log10(V_j_star), np.log10(St))

    # Calculate forward velocity index (m_theta)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
    m_theta = tables.get_forward_velocity_index(theta)

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    msap_j = Pi_star * A_j_star / (4 * np.pi * r_s_star ** 2) * d_function * f_function / (1 - M_0 * np.cos(np.pi / 180. * (theta - jet_delta))) * ((V_j_star - M_0) / V_j_star) ** m_theta

    # Multiply with number of engines
    # Normalize msap by reference pressure
    return msap_j * n_engines / p_ref**2


def jet_shock_source(V_j_star, M_j, A_j_star, Tt_j_star, theta, M_0, c_0, f, area_jet_effective, p_ref, n_shock, n_engines):
                     
    """
    Compute jet mixing noise mean-square acoustic pressure (msap).

    Parameters
    ----------
    V_j_star : 
    
    M_j : 
    
    A_j_star : 
    
    Tt_j_star : 
    
    M_0 : 
    
    c_0 : 

    f : np.ndarray
    
    area_jet_effective : 
    
    p_ref : 

    n_shock : 
    
    n_engines : int
    
    Returns
    -------
    np.ndarray : 

    """

    tables = JetShockNoiseTables()
    r_s_star = _R_SOURCE / np.sqrt(area_jet_effective)
    jet_delta = 0.

    # Calculate msap for all frequencies
    # If the jet is supersonic: shock cell noise
    if M_j > 1:
        # Calculate beta function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
        beta = (M_j ** 2 - 1) ** 0.5

        # Calculate eta (exponent of the pressure ratio parameter)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
        if beta > 1:
            if Tt_j_star < 1.1:
                eta = 1.
            else:
                eta = 2.
        else:
            eta = 4.

        # Calculate f_star
        # Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
        f_star = f * np.sqrt(area_jet_effective) / c_0

        # Calculate sigma parameter
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
        sigma = 7.80 * beta * (1 - M_0 * np.cos(np.pi / 180 * theta)) * np.sqrt(A_j_star) * f_star

        # Calculate W function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
        b = 0.23077
        W = 0
        for k in np.arange(1, n_shock):
            sum_inner = 0
            for m in np.arange(n_shock - k):
                # Calculate q_km
                q_km = 1.70 * k / V_j_star * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * V_j_star * np.cos(np.pi / 180 * theta))

                # Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
                sum_inner = sum_inner + np.sin((b * sigma * q_km / 2)) / (sigma * q_km) * np.cos(sigma * q_km)

            # Compute the correlation coefficient spectrum C
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Table II
            C = tables.get_correlation_coefficient_spectrum(np.log10(sigma))
            # C = 10**log10C

            # Add outer loop to the shock cell interference function
            W = W + (4. / (n_shock * b))* sum_inner * C ** (k ** 2)

        # Calculate the H function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
        log10H = tables.get_group_source_strength_spectrum(np.log10(sigma))

        # Source: Zorumski report 1982 part 2. Chapter 8.5.4
        if Tt_j_star < 1.1:
            log10H = log10H - 0.2
        H = (10 ** log10H)

        # Calculate mean-square acoustic pressure (msap)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
        msap_j = 1.92e-3 * A_j_star / (4 * np.pi * r_s_star ** 2) * (1 + W) / (1 - M_0 * np.cos(np.pi / 180. * (theta - jet_delta))) ** 4 * beta ** eta * H
    
    else:
        msap_j = np.zeros(f.size) * M_j ** 0

    # Normalize msap by reference pressure
    return msap_j * n_engines / p_ref**2
