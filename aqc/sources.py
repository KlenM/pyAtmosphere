import numpy as np
from dataclasses import dataclass

from aqc.theory.sources import GaussianBeam


@dataclass
class Source:
  wvl: float

  @property
  def k(self):
    return 2 * np.pi / self.wvl


class PlaneSource(Source):
  def output(self):
    return 1


class GaussianSource(GaussianBeam, Source):
  def output(self):
    return self.amplitude(self.channel.grid.get_rho2())


