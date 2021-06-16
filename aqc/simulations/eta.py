import numpy as np
from matplotlib import pyplot as plt
from aqc.pupils import CirclePupil
from aqc.simulations.simulation import OutputSimulation
from aqc.measures import eta


class EtaSimulation(OutputSimulation):
  def __init__(self, channel, radiuses, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = channel
    self.radiuses = radiuses
    self.pupils = [CirclePupil(radius=radius) for radius in radiuses]
    for pupil in self.pupils:
      pupil.channel = self.channel

    self._eta = np.zeros(len(radiuses))
    self._eta2 = np.zeros(len(radiuses))

  @property
  def eta(self):
    return self._eta / self.iteration

  @property
  def eta2(self):
    return self._eta2 / self.iteration

  @property
  def sigma_eta(self):
    return self._eta2 / self._eta**2 * self.iteration - 1
    
  def iter(self, *args, **kwargs):
    self.process(self.channel.run(pupil=False, *args, **kwargs))
  
  def process_output(self, output):
    etas = np.empty(len(self.radiuses))
    for i, pupil in enumerate(self.pupils):
      etas[i] = eta(self.channel, output=pupil.output(output))
    self._eta += etas
    self._eta2 += etas**2

  def print_output(self):
    print(f"Eta: {self.eta}")
    print(f"Sigma eta: {self.sigma_eta}")

  def plot_output(self):
    plt.figure(figsize=(10,6))
    plt.title("Залежність коефіцієнта проходження та апертурного сцинтиляційного індекса\nвід радіуса апертури")
    plt.plot(self.radiuses, self.eta, label='Середній коефіцієнт проходження $\\left< \eta \\right>$')
    plt.plot(self.radiuses, self.sigma_eta, label='Апертурний сцинтиляційний індекс $\\frac{\\left< \eta^2 \\right>}{\\left< \eta \\right>^2} - 1$')
    plt.xlabel(f"Aperture radius, $r$, м")
    plt.legend()
    plt.show()
