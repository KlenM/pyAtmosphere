import numpy as np
from scipy.integrate import quad


def get_rytov2(Cn2, k, length):
    return 1.23 * Cn2 * k**(7/6) * length**(11/6)


def get_r0(Cn2, k, length):
    """The coherence diameter for plane waves"""
    return (0.423 * k**2 * Cn2 * length)**(-3/5)


def get_r0s(Cn2, length, k):
    """The coherence diameter for spherical waves

    Parameters:
    Cn2 (float, list, function)
    length (float, list): Distance from source. List of lengths is using for providing the ends of intervals of values in Cn2 list.
    k (float): Wave vector of the source

    Returns:
    r0s (float): The coherence diameter for spherical waves"""

    # If Cn2 is a list of values
    try:
        _r0s = 0
        for i in range(len(Cn2)):
            try:
                length_i = length[i]
            except IndexError:
                length_i = length / len(Cn2)
            _r0s += get_r0s(Cn2[i], length_i, k)**(-5/3)
        return _r0s**(-3/5)
    except TypeError:
        pass

    # If Cn2 is a function
    try:
        _ = Cn2(length)

        def inintegral(z):
            return Cn2(z) * ((length - z) / length)**(5/3)
        return (0.423 * k**2 * quad(inintegral, 0, length)[0])**(-3/5)
    except TypeError:
        pass

    # If Cn2 is a constant
    return (0.423 * k**2 * Cn2 * 3/8 * length)**(-3/5)
