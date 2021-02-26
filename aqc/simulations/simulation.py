from IPython import display
from matplotlib import pyplot as plt


class Simulation:
  def __init__(self, print_skip=1, clear_plot=True):
    self.iteration = 0
    self.print_skip = print_skip
    self.clear_plot = clear_plot
    self.debug = None

  def iter(self, *args, **kwargs):
    raise NotImplementedError

  def print(self):
    raise NotImplementedError

  def final(self):
    raise NotImplementedError

  def run(self, *args, **kwargs):
    try:
      while True:
        self.iteration += 1
        self.iter(*args, **kwargs)
        if self.iteration % self.print_skip == 0:
          self._print()
    except KeyboardInterrupt:
      pass
    
    try:
      self._final()
    except NotImplementedError:
      self._print()

  def clear(self):
    if self.clear_plot:
      display.display(plt.gcf())
    display.clear_output(wait=True)
  
  def _print(self):
    self.clear()
    self.print()
    print(f"Iteration: {self.iteration}")
    if self.debug:
      print(f"Debug info: {self.debug}")
  
  def _final(self):
    self.clear()
    self.final()
    print(f"Iterations: {self.iteration}")
