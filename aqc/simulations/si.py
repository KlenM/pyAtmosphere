import numpy as np
from matplotlib import pyplot as plt
from aqc.simulations.simulation import PropagationSimulation
from aqc.theory.atmosphere.si import get_SI_andrews_strong


class SISimulation(PropagationSimulation):
  def __init__(self, channel, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = channel
    self._I_0 = np.zeros(shape=(len(channel.path.phase_screens) + 1))
    self._I_02 = np.zeros(shape=(len(channel.path.phase_screens) + 1))
    self.si_theory = get_SI_andrews_strong(self.positions, channel.path.phase_screen.model, channel.source)
  
  @property
  def positions(self):
    return np.array(list(self.channel.path.positions) + [self.channel.path.length])
  
  @property
  def si(self):
    return self._I_02 / self._I_0**2 * self.iteration - 1

  def iter(self, *args, **kwargs):
    for propagation_step, (propagation_result, _) in enumerate(self.channel.generator(pupil=False, store_output=True, *args, **kwargs)):
      self.process(propagation_result, propagation_step)
    self.process(self.channel.output, len(self.channel.path.phase_screens))
    self.iteration += 1
  
  def process_propagation(self, propagation_output, propagation_step):
    I_0 = abs(propagation_output[self.channel.grid.origin_index[0], self.channel.grid.origin_index[1]])**2
    self._I_0[propagation_step] += I_0
    self._I_02[propagation_step] += I_0**2
    
  def print_output(self):
    print(f"On-axis SI: {self.si}")

  def plot_output(self):
    plt.plot(self.positions, self.si, label=r"On-axis SI $\sigma_I$, m")
    plt.plot(self.positions, self.si_theory, label=r"Theoretical on-axis SI")
    plt.xlabel("Propagation distance z, m")
    plt.legend()
    plt.show()
  