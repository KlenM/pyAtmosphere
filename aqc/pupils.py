from dataclasses import dataclass

from aqc.aqc import AQCElement


@dataclass
class CirclePupil(AQCElement):
  radius: float

  def output(self, input):
    return input * (self.channel.grid.get_rho2() <= (self.radius)**2)