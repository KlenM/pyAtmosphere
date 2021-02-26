import cupy
import numpy as np
from aqc.utils import fft2, ifft2


def vacuum_propagation(input, length, k, delta, f2, f_delta):
  xp = cupy.get_array_module(input)
  return ifft2(xp.exp(1j * k * length) * xp.exp(-1j * xp.pi * length * (2 * xp.pi / k) * f2) * fft2(input, delta), f_delta)
