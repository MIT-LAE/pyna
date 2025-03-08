import numpy as np

_FREQUENCY_BAND_LOW = 17
_FREQUENCY_BAND_HIGH = 40


def get_frequency_bands(n_frequency_bands) -> None:
    """
    Compute the 1/3rd octave frequency bands.
        
    Parameters
    ----------
    n_frequency_bands: 
    
    Returns
    -------
    np.ndarray
        1/3rd octave frequencies
    """
    
    return 10 ** (0.1 * np.linspace(_FREQUENCY_BAND_LOW, _FREQUENCY_BAND_HIGH, n_frequency_bands))


def get_frequency_subbands(frequency, n_frequency_subbands):
    """
    Compute 1/3rd octave frequency sub-bands.

    Parameters
    ----------
    frequency : np.ndarray

    n_frequency_subbands : int
        Number of frequency sub-bands
    
    Returns
    -------
    np.ndarray
        Array of frequency sub-bands
    """

    # Calculate subband frequencies [Hz]
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 6-7
    # Source: Berton 2021 Simultaneous use of Ground Reflection and Lateral Attenuation Noise Models Appendix A Eq. 1
    
    n_frequency_bands = frequency.size

    frequency_subbands = np.zeros(n_frequency_subbands * n_frequency_bands)

    m = (n_frequency_subbands - 1) / 2.
    w = 2. ** (1 / (3. * n_frequency_subbands))
    
    for k in np.arange(n_frequency_bands):
        for h in np.arange(n_frequency_subbands):
            frequency_subbands[k * n_frequency_subbands + h] = w ** (h - m) * frequency[k]

    return frequency_subbands


def get_spectrum_subbands(msap, n_frequency_subbands):

    if not n_frequency_subbands%2:
        raise ValueError(
            f"Number of frequency subbands {n_frequency_subbands} is even;"
            "it has to be odd for symmetric around center frequency."
        )

    return _get_spectrum_subbands(msap, n_frequency_subbands)


def _get_spectrum_subbands(msap, n_frequency_subbands):
    """
    Compute subfrequency bands for a given 1/3rd octave frequency spectrum.

    Parameters
    ----------
    msap : np.ndarray

    n_frequency_subbands : int
    
    msap_in : np.ndarray
        Mean-square acoustic pressure of the source (re. rho_0,^2c_0^2) [-]
    
    Returns
    -------
    np.ndarray :
        Mean-square acoustic pressure of the source, split into frequency sub-bands
    """

    # Integer [-]
    n_frequency_bands = msap.size
    m = (n_frequency_subbands - 1) / 2

    # Initialize counter
    cntr = -1

    msap_sb = np.zeros(n_frequency_bands * n_frequency_subbands)

    for k in np.arange(n_frequency_bands):

        # Extract msap_k
        msap_in_k = msap[k]

        # Normalize the mean-square acoustic pressure: divide the vector by the average value
        # Computational fix for equations for u / v
        if sum(msap) == 0:
            msap_proc = msap
        else:
            msap_proc = msap / (np.sum(msap) / msap.shape[0])

        if msap_proc[k] == 0:
                msap_sb[k * n_frequency_subbands:(k + 1) * n_frequency_subbands] = (np.sum(msap) ** 0) * np.zeros(n_frequency_subbands)
        else:
            # Compute slope of spectrum
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 8-9
            if 0 < k < n_frequency_bands - 1:
                u = msap_proc[k] / msap_proc[k - 1]
                v = msap_proc[k + 1] / msap_proc[k]
            elif k == 0:
                u = msap_proc[1] / msap_proc[0]
                v = msap_proc[1] / msap_proc[0]
            elif k == n_frequency_bands - 1:
                u = msap_proc[k] / msap_proc[k - 1]
                v = msap_proc[k] / msap_proc[k - 1]

            # Compute constant A
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 12 + Berton ground-effects paper
            A = 1
            for h in np.arange(1, m + 1):
                A = A + u ** ((h - m - 1) / n_frequency_subbands) + v ** ((h) / n_frequency_subbands)

            # Compute MSAP in sub-bands
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 10 + Berton ground-effects paper
            for h in np.arange(n_frequency_subbands):
            
                cntr = cntr + 1

                if 0 <= h <= m - 1:
                    msap_sb[cntr] = (msap_in_k / A) * u ** ((h - m) / n_frequency_subbands)
                elif h == m:
                    msap_sb[cntr] = (msap_in_k / A)
                elif m + 1 <= h <= n_frequency_subbands - 1:
                    msap_sb[cntr] = (msap_in_k / A) * v ** ((h - m) / n_frequency_subbands)

    return msap_sb
