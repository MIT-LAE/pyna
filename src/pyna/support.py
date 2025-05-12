import numpy as np


def quantity_to_si(x):
    """Convert quantity to SI units.
    
    Parameters
    ----------
    x : Q_

    Returns
    -------
    float or np.ndarray
        Quantity magnitude in SI units.
    """

    if isinstance(x, np.ndarray):
        return [
            x_i.to_base_units().magnitude for x_i in x
        ]
    else:
        return x.to_base_units().magnitude