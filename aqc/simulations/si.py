import numpy as np
from matplotlib import pyplot as plt
from aqc.simulations.simulation import Simulation
from aqc.theory.atmosphere.si import get_SI_andrews_strong


class SISimulation(Simulation):
  def __init__(self, channel, print_skip=10, clear_plot=True):
    super().__init__(print_skip=print_skip, clear_plot=clear_plot)
    self.channel = channel
    self._I_0 = np.zeros(shape=(len(channel.path.phase_screens)))
    self._I_02 = np.zeros(shape=(len(channel.path.phase_screens)))

    self.si_theory = get_SI_andrews_strong(self.positions, channel.path.phase_screen.model, channel.source)
  
  @property
  def si(self):
    return self._I_02 / self._I_0**2 * self.iteration - 1
  
  @property
  def positions(self):
    positions = np.empty(shape=(len(self.channel.path.phase_screens)))
    for i, _ in enumerate(positions):
      if i == 0:
        positions[i] = self.channel.path.phase_screens[i].thickness
      else:
        positions[i] = positions[i - 1] + self.channel.path.phase_screens[i].thickness
    return positions

  def iter(self, *args, **kwargs):
    I_0 = abs(self.channel.run(pupil=False, return_intermediate=True, *args, **kwargs)[:, self.channel.grid.origin_index[0], self.channel.grid.origin_index[1]].get())**2
    self._I_0 += I_0
    self._I_02 += I_0**2
  
  def print(self):
    plt.plot(self.positions, self.si, label=r"On-axis SI $\sigma_I$, m")
    plt.plot(self.positions, self.si_theory, label=r"Theoretical on-axis SI")
    plt.xlabel("Propagation distance z, m")
    plt.legend()
    plt.show()
  