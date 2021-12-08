from dataclasses import dataclass
from typing import Sequence

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


class Default:
    def __init__(self, default_path):
        self._default_path_list = default_path.split(".")

    def __get__(self, obj, cls):
        current_node = obj
        for node in self._default_path_list:
            current_node = getattr(current_node, node)
        return current_node


@dataclass
class PolarDiscreteFunction():
    rho: Sequence[float]
    theta: Sequence[float]
    value: Sequence[float]


def fft2(x, delta):
    xp = get_xp()
    return xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(x))) * delta**2


def ifft2(x, delta):
    xp = get_xp()
    N = x.shape[0]
    return xp.fft.ifftshift(xp.fft.ifft2(xp.fft.ifftshift(x))) * (N * delta)**2
