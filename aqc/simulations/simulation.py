from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence
from IPython import display
from matplotlib import pyplot as plt


class Simulation(ABC):
  iteration: int = 0

  def __init__(
      self, 
      print_step: int = 0, 
      plot_step: int = 0, 
      save_step: int = 0, 
      save_path: str = "", 
      output_function: Callable[[Simulation], None] = None
      ):
    self.print_step = print_step
    self.plot_step = plot_step
    self.save_step = save_step
    self.save_path = save_path
    self.output_function = output_function

  def run(self, *args, **kwargs):
    try:
      while True:
        self.iter(*args, **kwargs)
        self.iteration += 1
        self.output()
    except KeyboardInterrupt:
      pass
    finally:
      self.output()

  @abstractmethod
  def iter(self, *args, **kwargs):
    pass

  def save_output(self):
    raise NotImplementedError

  def plot_output(self):
    raise NotImplementedError

  def print_output(self):
    raise NotImplementedError

  def output(self):
    if self.output_function:
      self.output_function()
    if self.save_step and self.iteration % self.save_step == 0:
      self.save_output()

    is_plot_iteration = self.plot_step and self.iteration % self.plot_step == 0
    is_print_iteration = self.print_step and self.iteration % self.print_step == 0

    if is_plot_iteration or is_print_iteration:
      if is_plot_iteration:
        self.plot_output()
      if is_print_iteration:
        self.print_output()
      print(f"Iteration: {self.iteration}")
      display.clear_output(wait=True)


class MultipleSimulation(Simulation):
  pass


class OutputSimulation(Simulation):
  @abstractmethod
  def process_output(self, output):
    pass


class PupilSimulation(Simulation):
  @abstractmethod
  def process_pupil_output(self, pupil_output):
    pass


class PropagationSimulation(Simulation):
  @abstractmethod
  def process_propagation(self, propagation_result, propagation_step):
    pass


class PhaseScreenSimulation(Simulation):
  @abstractmethod
  def process_phase_screen(self, phase_screen):
    pass
