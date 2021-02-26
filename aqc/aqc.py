from dataclasses import dataclass
import cupy
import numpy as np


class AQCElement:
  def set_channel(self, channel):
    self.channel = channel


@dataclass
class AQC:
  grid: AQCElement
  source: AQCElement
  path: AQCElement
  pupil: AQCElement = None
  f_grid: AQCElement = None
  use_GPU: bool = True
  
  def __post_init__(self):    
    self.grid.xp = cupy if self.use_GPU else np
    self.source.set_channel(self)
    self.path.set_channel(self)
    if self.pupil: 
      self.pupil.set_channel(self)
    if self.f_grid:
      self.f_grid.xp = cupy if self.use_GPU else np

  def run(self, pupil=True, *args, **kwargs):
    if pupil:
      return self.pupil.output(self.path.output(self.source.output(), *args, **kwargs))    
    return self.path.output(self.source.output(), *args, **kwargs)
  