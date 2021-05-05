import numpy as np
from matplotlib import pyplot as plt
from aqc.simulations.simulation import Simulation, SimulationGUI
from aqc.measures import eta


class PDTSimulation(Simulation):
  type = "pupil"
  
  def __init__(self, channel, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = channel
    self.etas = []
  
  def process(self, pupil_output):
    self.etas.append(eta(self.channel, output=pupil_output))
  
  def iter(self, *args, **kwargs):
    self.process(self.channel.run(pupil=True))
    self.iteration += 1
  
  def print(self):
    print(f"Etas: {self.etas}")


class PDTSimulationGUI(PDTSimulation, SimulationGUI):
  def print(self):
    plt.hist(self.etas, bins=25, range=(0, 1), density=True)
    plt.ylabel(r"Probability distribution $\mathcal{P}\,(\eta)$")
    plt.xlabel(f"Transmittatance, $\eta$")
    plt.show()