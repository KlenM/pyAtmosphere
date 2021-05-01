from dataclasses import dataclass


@dataclass
class CirclePupil:
  radius: float

  def output(self, input):
    return input * (self.channel.grid.get_rho2() <= (self.radius)**2)