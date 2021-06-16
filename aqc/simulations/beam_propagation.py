import numpy as np
import cupy
from matplotlib import pyplot as plt

from aqc.simulations.simulation import PropagationSimulation
from aqc.measures import mean_r, mean_r2
from aqc.theory.atmosphere.beam_wandering import get_r_bw
from aqc.theory.atmosphere.long_term import get_numeric_w_LT


class BeamPropagationSimulation(PropagationSimulation):  
  def __init__(self, channel, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = channel
    self._bw2 = np.zeros(shape=(len(channel.path.phase_screens) + 1))
    self._lt2 = np.zeros(shape=(len(channel.path.phase_screens) + 1))
    self._st2 = np.zeros(shape=(len(channel.path.phase_screens) + 1))

    self.bw_theoretical = [get_r_bw(L, self.channel.path.phase_screen.model, self.channel.source) for L in self.positions]
    rho = cupy.asnumpy(self.channel.grid.get_x()[0, self.channel.grid.origin_index[0]::4])
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
    return np.array(list(self.channel.path.positions) + [self.channel.path.length])

  def iter(self, *args, **kwargs):
    for propagation_step, (propagation_result, _) in enumerate(self.channel.generator(pupil=False, store_output=True, *args, **kwargs)):
      self.process_propagation(propagation_result, propagation_step)
    self.process_propagation(self.channel.output, len(self.channel.path.phase_screens))

  def process_propagation(self, propagation_result, propagation_step):
    r = mean_r(self.channel, output=propagation_result)
    r2 = mean_r2(self.channel, output=propagation_result)
    self._bw2[propagation_step] += r**2
    self._lt2[propagation_step] += 2 * r2
    self._st2[propagation_step] += 2 * r2 - r**2
  
  def print(self):
    print(f"Beam wandering: {self.bw}")
    print(f"Lont term: {self.lt}")
    print(f"Short term: {self.st}")

  def plot_output(self):
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
