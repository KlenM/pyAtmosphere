import numpy as np
from typing import List, Sequence
from pathlib import Path
from uuid import uuid4
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from aqc.simulations.simulation import Simulation
from aqc.measures import eta


class TimeCoherenceSimulation(Simulation):
  def __init__(self, channel, time: Sequence, wind_speed: float = 1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if time[0] != 0:
      raise ValueError("The first element of the time sequence should be equal 0")
    self.channel = channel
    self.time = time
    self.wind_speed = wind_speed
    self.etas : List[List[float]] = [[] for _ in self.time]
    self.simulation_uuid = uuid4()
    
  def iter(self):
    for ps in self.channel.path.phase_screens:
      ps._cached_spectrum = None
    for i, t in enumerate(self.time):
        self.etas[i].append(eta(self.channel, shift=(t * self.wind_speed, 0), use_cached_spectrum=True))

  @property
  def tc(self) -> Sequence[float]:
    return [pearsonr(self.etas[0], etas_i)[0] for etas_i in self.etas]

  def plot_output(self):
    if len(self.etas[0]) > 2:
      plt.plot(self.time, self.tc)
    plt.show()
  
  def save_output(self):
    np.save(Path(self.save_path) / f"{self.simulation_uuid}.npy", np.asarray(self.etas, dtype=np.float32), allow_pickle=False)


class TimeCoherenceVsl0Simulation(Simulation):
  def __init__(self, channels, time: Sequence, wind_speed: float = 1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.simulations = [TimeCoherenceSimulation(channel=channel, time=time, wind_speed=wind_speed) for channel in channels]
    self.params = [s.channel.path.phase_screen.model.l0 for s in self.simulations]
    self.simulation_uuid = uuid4()
  
  def iter(self):
    for simulation in self.simulations:
      simulation.iter()

  def plot_output(self):
    if self.iteration > 2:
        plt.plot(self.params, [s.tc[1:] for s in self.simulations])
    plt.show()
  
  def save_output(self):
    data = np.asarray([simulation.etas for simulation in self.simulations], dtype=np.float32)
    np.save(Path(self.save_path) / f"{self.simulation_uuid}.npy", data, allow_pickle=False)

