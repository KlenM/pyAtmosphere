import cupy
import numpy as np
from matplotlib import pyplot as plt

from aqc.simulations.simulation import Simulation, SimulationGUI
from aqc.theory.phase_screens.sf import calculate_sf


class SFSimulation(Simulation):
  type = "phase_screen"
  
  def __init__(self, channel, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = channel
    
    xp = self.channel.grid.get_array_module()
    phase_screen_delta = self.channel.path.phase_screen.grid.delta
    phase_screen_resolution = self.channel.path.phase_screen.grid.resolution[0]
    self.phase_screen_generator = self.channel.path.phase_screen.generator()
    self._sf_unnormed = xp.zeros((phase_screen_resolution - 1, ))
    self.r = np.linspace(0, phase_screen_delta * phase_screen_resolution, phase_screen_resolution)[:-1]
    # self.theoretical_sf = self.get_theoretical_sf()
    self.numerical_theoretical_sf = self.get_numerical_theoretical_sf()

  @property
  def sf(self):
    return self._sf_unnormed / self.iteration

  def get_theoretical_sf(self):
    k = 2 * np.pi / self.channel.path.phase_screen.wvl
    return self.channel.path.phase_screen.model.sf_phi(self.r, k, self.channel.path.phase_screen.thickness)

  def get_numerical_theoretical_sf(self):
    k = 2 * np.pi / self.channel.path.phase_screen.wvl
    return self.channel.path.phase_screen.model.sf_phi_numeric(self.r, k, self.channel.path.phase_screen.thickness)

  def iter(self, *args, **kwargs):
    self.process(next(self.phase_screen_generator))
    self.iteration += 1
  
  def process(self, phase_screen, propagation_step=None):
    self._sf_unnormed[:] = self._sf_unnormed + calculate_sf(phase_screen).mean(axis=1)

  def print(self):
    print("Mean square error = {NOT IMPLEMENTED}")
    print(self.sf)
  

class SFSimulationGUI(SFSimulation, SimulationGUI):
  def print(self):
    plt.plot(self.r, cupy.asnumpy(self.sf), label="Simulated")
    # plt.plot(self.r, cupy.asnumpy(self.theoretical_sf), label="Theoretical")
    plt.plot(self.r, cupy.asnumpy(self.numerical_theoretical_sf), label="Theoretical numerical")
    ## ToDo: Axis!
    plt.legend()
    plt.show()