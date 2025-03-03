import numpy as np
from pyna.noise_tables import (
    PerceivedNoiseTables
)

def compute_spl(msap, rho_0, c_0):
    """
    Compute sound pressure level.

    Parameters
    ----------
    msap : np.ndarray
        Mean-square acoustic pressure [-]
    rho_0 : float
        Ambient density [kg/m3]
    c_0 : float 
        Ambient speed of sound [m/s]
    
    Returns
    -------
    np.ndarray
        Sound pressure levels for 1/3rd octave sub-band frequencies
    
    """

    # Allocate
    spl = np.zeros_like(msap)

    # Index variable where msap > 0
    idx_msap_positive = msap > 0
    spl[idx_msap_positive] = _compute_spl(msap[idx_msap_positive], rho_0, c_0)

    return spl

def _compute_spl(msap, rho_0, c_0):

    return 10*np.log10(msap) + 20.*np.log10(rho_0 * c_0 ** 2.)


def compute_oaspl(spl):
    """Compute overall sound pressure level.

    Parameters
    ----------
    spl : np.ndarray
        sound pressure level [dB]

    Returns
    -------
    np.ndarray

    """

    # Compute OASPL by summing SPL logarithmically
    return 10 * np.log10(np.sum(10 ** (spl / 10.)))

def compute_noy(spl):

    """Compute noy.
    
    Parameters
    ----------
    spl : np.ndararray
        
    Returns
    -------
    np.ndarray
        Noy
    """

    tables = PerceivedNoiseTables()

    # Source: ICAO Annex 16 Appendix 2 section 4.2 Step 1
    noy = np.zeros(spl.size)
    
    for i in np.arange(spl.size):
        if tables.get_noy_spl_a(i) <= spl[i]:
            noy[i] = 10 ** (tables.get_noy_m_c(i) * (spl[i] - tables.get_noy_spl_c(i)))
        
        elif tables.get_noy_spl_b(i) <= spl[i] <= tables.get_noy_spl_a(i):
            noy[i] = 10 ** (tables.get_noy_m_b(i) * (spl[i] - tables.get_noy_spl_b(i)))
        
        elif tables.get_noy_spl_e(i) <= spl[i] <= tables.get_noy_spl_b(i):
            noy[i] = 0.3 * 10 ** (tables.get_noy_m_e(i) * (spl[i] - tables.get_noy_spl_e(i)))
        
        elif tables.get_noy_spl_d(i) <= spl[i] <= tables.get_noy_spl_e(i):
            noy[i] = 0.1 * 10 ** (tables.get_noy_m_d(i) * (spl[i] - tables.get_noy_spl_d(i)))
        
        else:
            noy[i] = 0.

    return noy

def compute_pnl(noy):
    """_summary_

    Parameters
    ----------
    noy : np.ndarray

    Returns
    -------
    float:
        Perceived noise level (pnl)
        
    """

    n_t = np.max(noy) + 0.15 * (np.sum(noy) - np.max(noy))

    return 40 + 10. / np.log10(2) * np.log10(n_t)

def compute_tonal_corrections(spl):

    # Spectral irregularities correction
    # Step 1: Compute the slope of SPL
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 1
    s = np.zeros(spl.size)
    for i in np.arange(spl.size):
        # Up to the 3th band the table has no value
        if i <= 2:
            s[i] = np.nan
        # Start at the 4th band
        else:
            s[i] = spl[i] - spl[i - 1]

    # Step 2: Compute the absolute value of the slope and compare to 5
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 2
    slope = np.zeros(spl.size)
    slope_large = np.zeros(spl.size)
    for i in np.arange(spl.size):
        # Compute the absolute value of the slope
        slope[i] = s[i] - s[i - 1]
        # Check if slope is larger than 5
        if abs(slope[i]) > 5:
            slope_large[i] = 1

    # Step 3: Compute the encircled values of SPL
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 3
    spl_large = np.zeros(spl.size)
    for i in np.arange(spl.size):
        # Check if value of slope is encircled
        if slope_large[i] == 1:
            # Check if value of slope is positive and greater than previous slope
            if s[i] > 0 and s[i] > s[i - 1]:
                spl_large[i] = 1
            elif s[i] <= 0 < s[i - 1]:
                spl_large[i - 1] = 1

    # Step 4: Compute new adjusted sound pressure levels SPL'
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 4
    spl_p = np.zeros(spl.size)
    for i in np.arange(spl.size):
        if spl_large[i] == 0:
            spl_p[i] = spl[i]
        elif spl_large[i] == 1:
            if i <= 22:
                spl_p[i] = 0.5 * (spl[i - 1] + spl[i + 1])
            elif i == 23:
                spl_p[i] = spl[22] + s[22]

    # Step 5: Recompute the slope s'
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 5
    s_p = np.zeros(spl.size + 1)
    for i in np.flip(np.arange(spl.size), 0):
        # From 4th band onwards
        if i > 2:
            s_p[i] = spl_p[i] - spl_p[i - 1]
        # At the 3rd band
        elif i == 2:
            s_p[i] = s_p[i + 1]
    # Compute 25th imaginary band
    s_p[24] = s_p[23]

    # Step 6: Compute arithmetic average of the 3 adjacent slopes
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 6
    s_bar = np.zeros(spl.size)
    for i in np.arange(2, spl.size - 1):
        s_bar[i] = 1. / 3. * (s_p[i] + s_p[i + 1] + s_p[i + 2])

    # Step 7: Compute final 1/3 octave-band sound pressure level
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 7
    spl_pp = np.zeros(spl.size)
    for i in np.arange(2, spl.size):
        if i == 2:
            spl_pp[i] = spl[i]
        elif i > 2:
            spl_pp[i] = spl_pp[i - 1] + s_bar[i - 1]

    # Step 8: Compute the difference between SPL and SPL_pp
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 8
    F = np.zeros(spl.size)
    for i in np.arange(spl.size):
        # Compute the difference and limit at 1.5
        F[i] = spl[i] - spl_pp[i]
        # Check values larger than 1.5 (ICAO Appendix 2-16)
        if F[i] < 1.5:
            F[i] = 0

    # Step 9: Compute the correction factor C
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 9
    tone_corrections = np.zeros(spl.size)
    for i in np.arange(2, spl.size):
        if i < 10:  # Frequency in [50,500[
            if 1.5 <= F[i] < 3:
                tone_corrections[i] = F[i] / 3. - 0.5
            elif 3. <= F[i] < 20.:
                tone_corrections[i] = F[i] / 6.
            elif F[i] >= 20.:
                tone_corrections[i] = 3. + 1 / 3.
        elif 10 <= i <= 20:  # Frequency in [500,5000]
            if 1.5 <= F[i] < 3.:
                tone_corrections[i] = 2. * F[i] / 3. - 1.0
            elif 3 <= F[i] < 20:
                tone_corrections[i] = F[i] / 3.
            elif F[i] >= 20:
                tone_corrections[i] = 6. + 2. / 3.
        elif i > 20:  # Frequency in ]5000,10000]
            if 1.5 <= F[i] < 3.:
                tone_corrections[i] = F[i] / 3. - 0.5
            elif 3. <= F[i] < 20.:
                tone_corrections[i] = F[i] / 6.
            elif F[i] >= 20.:
                tone_corrections[i] = 3. + 1. / 3.
    
    return tone_corrections
    
def compute_pnlt(spl, flag_tones_under_800Hz):
    """Compute perceived noise level, tone corrected [PNdB]

    Parameters
    ----------
    spl : np.ndarray
        sound pressure level [dB]
    flag_tones_under_800Hz : bool

    
    Returns
    -------
    np.ndarray
    
    """

    # Source: ICAO Annex 16 Appendix 2 Table A2-3 (Noy tables)
    noy = compute_noy(spl)

    # Source: ICAO Annex 16 Appendix 2 section 4.2 Step 2-3
    pnl = compute_pnl(noy)

    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 1-9
    tone_corrections = compute_tonal_corrections(spl)    

    # Step 10: Compute the largest of the tone correction
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 10
    if not flag_tones_under_800Hz:
        c_max = np.max(tone_corrections[13:])
    else:
        c_max = np.max(tone_corrections)

    # Compute tone-corrected perceived noise level (pnlt)
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 10
    return pnl + c_max
