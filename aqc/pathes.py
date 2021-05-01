from dataclasses import dataclass
import cupy
import numpy as np
import copy

from aqc.aqc import config
from aqc.theory.vacuum import vacuum_propagation


@dataclass
class VacuumPath:
  length: float

  def output(self, input, length=None):
    length = length if not length is None else self.length
    return vacuum_propagation(
      input, 
      length,
      self.channel.source.k, 
      self.channel.grid.delta,
      self.channel.grid.get_f_grid().get_rho2(), 
      self.channel.grid.get_f_grid().delta
      )


@dataclass
class PhaseScreenPath:
  phase_screens: object
  losses_db: float = 0

  def init_phase_screens(self):
    for phase_screen in self.phase_screens:
      phase_screen.channel = self.channel

  def output(self, input, return_intermediate=False):
    self.init_phase_screens()

    xp = cupy.get_array_module(input)
    sg = xp.exp(-self.channel.grid.get_N2()**8 / (0.47 * np.min(self.channel.grid.resolution))**16, dtype=config["dtype"]["float"])
    
    intermediate_results = []
    for phase_screen in self.phase_screens:
      vacuum_path = VacuumPath(phase_screen.thickness)
      vacuum_path.channel = self.channel
      input = vacuum_path.output(sg * xp.exp(-1j * phase_screen.generate()) * input).astype(config["dtype"]["complex"])

      if return_intermediate:
        intermediate_results.append(input.copy())
    
    if return_intermediate:
      return self.append_losses(xp.array(intermediate_results))
    return self.append_losses(input)

  def append_losses(self, input):
    if self.losses_db:
      return input * 10**(-self.losses_db / 20)
    return input


class IdenticalPhaseScreensPath(PhaseScreenPath):
  def __init__(self, phase_screen, length, count, losses_db=0):
    self.phase_screen = phase_screen
    self._length = length
    self._count = count

    phase_screen.thickness = length / count
    super().__init__([copy.copy(self.phase_screen) for i in range(count)], losses_db=losses_db)
    self.phase_screen = self.phase_screens[0]
  
  @property
  def length(self):
      return self._length

  @length.setter
  def length(self, value):
    self._length = value
    self.init_phase_screens()

  @property
  def count(self):
      return self._count

  @count.setter
  def count(self, value):
    self._count = value
    self.init_phase_screens()