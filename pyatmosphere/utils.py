import numpy as np

from pyatmosphere.gpu import get_xp


class CrossRef:
    def __init__(self, cross_ref_name):
        self.cross_ref_name = cross_ref_name

    def __set_name__(self, obj, name):
        self.privat_name = "_" + name

    def __get__(self, obj, type=None):
        return getattr(obj, self.privat_name)

    def __set__(self, obj, value):
        if not value:
            return
        setattr(obj, self.privat_name, value)
        setattr(value, self.cross_ref_name, obj)


def fft2(x, delta):
    xp = get_xp()
    return xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(x))) * delta**2


def ifft2(x, delta):
    xp = get_xp()
    N = x.shape[0]
    return xp.fft.ifftshift(xp.fft.ifft2(xp.fft.ifftshift(x))) * (N * delta)**2
