import numpy as np

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

    # Compute SPL
    spl = 10*np.log10(msap) + 20.*np.log10(rho_0 * c_0 ** 2.)

    # Clip values to avoid (spl < 0)
    return spl.clip(min=1e-99)

