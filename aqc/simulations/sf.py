import cupy
import numpy as np
from matplotlib import pyplot as plt

from aqc.simulations.simulation import Simulation
from aqc.theory.phase_screens.sf import calculate_sf


class SFSimulation(Simulation):
  def __init__(self, phase_screen, print_skip=1, clear_plot=True):
    super().__init__(print_skip=print_skip, clear_plot=clear_plot)
    self.phase_screen = phase_screen
    self.ps_generator = phase_screen.generator()
    self.sfs_unnormed = phase_screen.grid.xp.zeros((phase_screen.grid.resolution[0] - 1, ))
    self.r = np.linspace(0, phase_screen.grid.delta * phase_screen.grid.resolution[0], phase_screen.grid.resolution[0])[:-1]
    self.theoretical_sf = self.get_theoretical_sf()
    self.numerical_theoretical_sf = self.get_numerical_theoretical_sf()

  @property
  def sf(self):
    return self.sfs_unnormed / self.iteration

  def get_theoretical_sf(self):
    k = 2 * np.pi / self.phase_screen.wvl
    return self.phase_screen.model.sf_phi(self.r, k, self.phase_screen.thickness)

  def get_numerical_theoretical_sf(self):
    k = 2 * np.pi / self.phase_screen.wvl
    return self.phase_screen.model.sf_phi_numeric(self.r, k, self.phase_screen.thickness)

  def iter(self, *args, **kwargs):
    self.sfs_unnormed[:] = self.sfs_unnormed + calculate_sf(next(self.ps_generator)).mean(axis=1)

  def print(self):
    plt.plot(self.r, cupy.asnumpy(self.sf), label="Simulated")
    # plt.plot(self.r, cupy.asnumpy(self.theoretical_sf), label="Theoretical")
    plt.plot(self.r, cupy.asnumpy(self.numerical_theoretical_sf), label="Theoretical numerical")
    plt.legend()
    plt.show()
  