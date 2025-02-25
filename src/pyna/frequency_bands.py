import numpy as np

_L_I = 16

def compute_frequency_bands(n_frequency_bands) -> None:
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
    
    return 10 ** (0.1 * np.linspace(_L_I + 1, 40, n_frequency_bands))


def compute_frequency_subbands(f, n_frequency_subbands):
    """
    Compute 1/3rd octave frequency sub-bands.

    Parameters
    ----------
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
    
    n_frequency_bands = f.size

    f_sb = np.zeros(n_frequency_subbands * n_frequency_bands)

    m = (n_frequency_subbands - 1) / 2.
    w = 2. ** (1 / (3. * n_frequency_subbands))
    
    for k in np.arange(n_frequency_bands):
        for h in np.arange(n_frequency_subbands):
            f_sb[k * n_frequency_subbands + h] = w ** (h - m) * f[k]

    return f_sb