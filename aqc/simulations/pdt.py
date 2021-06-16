import numpy as np
from matplotlib import pyplot as plt
from aqc.simulations.simulation import OutputSimulation
from aqc.measures import eta


class PDTSimulation(OutputSimulation):
  def __init__(self, channel, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = channel
    self.etas = []
  
  def iter(self, *args, **kwargs):
    self.process_output(self.channel.run(pupil=True))
  
  def process_output(self, output):
    self.etas.append(eta(self.channel, output=output))
  
  def print_output(self):
    print(f"Etas: {self.etas}")
    
  def plot_output(self):
    plt.hist(self.etas, bins=25, range=(0, 1), density=True)
    plt.ylabel(r"Probability distribution $\mathcal{P}\,(\eta)$")
    plt.xlabel(f"Transmittatance, $\eta$")
    plt.show()
