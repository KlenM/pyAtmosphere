from pyatmosphere.gpu import get_xp
from pyatmosphere.utils import fft2, ifft2


def vacuum_propagation(input, length, k, delta, f2, f_delta):
    xp = get_xp()
    return ifft2(xp.exp(1j * k * length) * xp.exp(-1j * xp.pi * length * (2 * xp.pi / k) * f2) * fft2(input, delta), f_delta)
