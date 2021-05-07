import sys
from IPython import display
from matplotlib import pyplot as plt
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class Simulation:
  type: Literal["output", "propagation", "phase_screen"]
  iteration: int = 0

  def iter(self, *args, **kwargs):
    self.process(*args, **kwargs)

  def process(self, *args, **kwargs):
    self.iteration += 1
    raise NotImplementedError

  def print(self):
    raise NotImplementedError

  def final(self):
    raise NotImplementedError

  def run(self, *args, **kwargs):
    try:
      while True:
        self.iter(*args, **kwargs)
        self._print()
    except KeyboardInterrupt:
      pass
    finally:
      self._final()

  def _print(self):
    self.print()
    print(f"Iteration: {self.iteration}")
    display.clear_output(wait=True)
  
  def _final(self):
    try:
      self.final()
    except NotImplementedError:
      self.print()
    print(f"Iterations: {self.iteration}")


class SimulationGUI(Simulation):
  def __init__(self, plot_skip=1, *args, **kwargs):
    self.plot_skip = plot_skip
    super().__init__(*args, **kwargs)
  
  def _print(self):
    if self.iteration % self.plot_skip == 0:
      self.print()
      print(f"Iteration: {self.iteration}")
      display.display(plt.gcf())
      display.clear_output(wait=True)