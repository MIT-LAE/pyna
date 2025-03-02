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

    # Allocate
    spl = np.zeros_like(msap)

    # Index variable where msap > 0
    idx_msap_positive = msap > 0
    spl[idx_msap_positive] = _compute_spl(msap[idx_msap_positive], rho_0, c_0)

    return spl

def _compute_spl(msap, rho_0, c_0):

    return 10*np.log10(msap) + 20.*np.log10(rho_0 * c_0 ** 2.)
