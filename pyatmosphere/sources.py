import numpy as np
from typing import Union, Sequence, Optional
from dataclasses import dataclass

from pyatmosphere.gpu import get_xp
from pyatmosphere.theory.sources import GaussianBeam


@dataclass
class Source:
    wvl: float

    @property
    def k(self):
        return 2 * np.pi / self.wvl


class PlaneSource(Source):
    def __init__(self, wvl, angle: Union[float, Sequence[float]] = 0, phase: Optional[Sequence[float]] = None):
        self.angle = [angle] if isinstance(angle, float) else angle
        self.phase = [0] * len(self.angle) if phase is None else phase
        if len(self.angle) != len(self.phase):
            raise ValueError("Length of 'angle' sequence must be equal to the length of 'phase'.")
        super().__init__(wvl)

    def output(self):
        xp = get_xp()
        angle = np.asarray(self.angle).reshape(-1, 1, 1)
        phase = np.asarray(self.phase).reshape(-1, 1, 1)
        phase = self.k * np.sin(angle) * self.channel.grid.get_x() - phase
        ui = xp.exp(- 1j * phase).mean(axis=0)
        return ui * xp.ones_like(ui).T


class GaussianSource(GaussianBeam, Source):
    def output(self):
        return self.amplitude(self.channel.grid.get_rho2())
