import numpy as np
from matplotlib import pyplot as plt

from aqc.simulations.simulation import Simulation
from aqc.measures import mean_r, mean_r2
from aqc.theory.atmosphere.beam_wandering import get_r_bw
from aqc.theory.atmosphere.long_term import get_numeric_w_LT


class BeamPropagationSimulation(Simulation):
  def __init__(self, channel, print_skip=30, clear_plot=True):
    super().__init__(clear_plot=clear_plot, print_skip=print_skip)
    self.channel = channel
    self._bw2 = np.zeros(shape=(len(channel.path.phase_screens)))
    self._lt2 = np.zeros(shape=(len(channel.path.phase_screens)))
    self._st2 = np.zeros(shape=(len(channel.path.phase_screens)))

    self.bw_theoretical = [get_r_bw(L, self.channel.path.phase_screen.model, self.channel.source) for L in self.positions]
    rho = self.channel.grid.get_x()[0, self.channel.grid.origin_index[0]::4].get()
    w_LT = lambda L: get_numeric_w_LT(L, self.channel.path.phase_screen.model, self.channel.source.w0, self.channel.source.wvl, self.channel.source.F0, rho, self.channel.grid.delta)
    self.lt_theoretical = [w_LT(L) for L in self.positions]
  
  @property
  def bw(self):
    return np.sqrt(self._bw2 / self.iteration)
    
  @property
  def lt(self):
    return np.sqrt(self._lt2 / self.iteration)
    
  @property
  def st(self):
    return np.sqrt(self._st2 / self.iteration)
  
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
    r = mean_r(self.channel, return_intermediate=True, *args, **kwargs)
    r2 = mean_r2(self.channel, return_intermediate=True, *args, **kwargs)
    
    self._bw2 += r**2
    self._lt2 += 2 * r2
    self._st2 += 2 * r2 - r**2
  
  # def print(self):
  #   plt.plot(self.positions, self.bw, label=r"Beam wandering $\left<r_c\right>$, m")
  #   plt.plot(self.positions, self.lt, label=r"Long term $W_{LT}$, m")
  #   plt.plot(self.positions, self.st, label=r"Short term $W_{ST}$, m")
  #   plt.xlabel("Propagation distance z, m")
  #   plt.legend()
  #   plt.show()
  
  def print(self):
    fig, axs = plt.subplots(3, 1, figsize=(8,9))
    axs[0].set_ylabel(r"Beam wandering $\left<r_c\right>$, m")
    axs[0].plot(self.positions, self.bw, label="Simulated")
    axs[0].plot(self.positions, self.bw_theoretical, label="Theory")
    axs[0].legend()

    axs[1].set_ylabel(r"Long term $W_{LT}$, m")
    axs[1].plot(self.positions, self.lt, label="Simulated")
    axs[1].plot(self.positions, self.lt_theoretical, label="Theory")
    axs[1].legend()

    axs[2].set_ylabel(r"Short term $W_{ST}$, m")
    axs[2].plot(self.positions, self.st)
    axs[2].set_xlabel("Propagation distance z, m")
    plt.show()