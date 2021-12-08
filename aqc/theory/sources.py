import numpy as np

from aqc.gpu import get_xp


class GaussianBeam:
    def __init__(self, wvl, w0, F0):
        self.wvl = wvl
        self.w0 = w0
        self.F0 = F0

    @property
    def k(self):
        return 2 * np.pi / self.wvl

    def amplitude(self, r2):
        xp = get_xp()
        return xp.sqrt(2 / xp.pi) / self.w0 * xp.exp(-(1 / self.w0**2 + 1j * 2 * xp.pi / self.wvl / 2 / self.F0) * r2)

    def get_theta0(self, length):
        return 1 - length / self.F0

    def get_Lambda0(self, length):
        return 2 * length / self.k / self.w0**2

    def get_theta(self, length):
        return self.get_theta0(length) / (self.get_theta0(length)**2 + self.get_Lambda0(length)**2)

    def get_Lambda(self, length):
        return self.get_Lambda0(length) / (self.get_theta0(length)**2 + self.get_Lambda0(length)**2)

    def get_w(self, length):
        return self.w0 * np.sqrt(self.get_theta0(length)**2 + self.get_Lambda0(length)**2)
