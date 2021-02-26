import numpy as np
from matplotlib import pyplot as plt
from aqc.simulations.simulation import Simulation
from aqc.measures import eta


class PDTSimulation(Simulation):
  def __init__(self, channel, print_skip=5, clear_plot=True):
    super().__init__(print_skip=print_skip, clear_plot=print_skip)
    self.channel = channel
    self.etas = []
  
  def iter(self, *args, **kwargs):
    self.etas.append(eta(self.channel, *args, **kwargs))
  
  def print(self):
    plt.hist(self.etas, bins=25, range=(0, 1), density=True)
    plt.ylabel(r"Probability distribution $\mathcal{P}\,(\eta)$")
    plt.xlabel(f"Transmittatance, $\eta$")
    plt.show()