from dataclasses import dataclass
import numpy as np
import cupy


config = {
  "use_gpu": True,
  "dtype": {
    "float": np.float32,
    "complex": np.complex64
  }
}


class AQCElement:
  def set_channel(self, channel):
    self.channel = channel


@dataclass
class AQC:
  grid: AQCElement
  source: AQCElement
  path: AQCElement
  pupil: AQCElement = None
  use_GPU: bool = True
  
  def __post_init__(self):
    self.source.set_channel(self)
    self.path.set_channel(self)
    if self.pupil: 
      self.pupil.set_channel(self)

  def run(self, pupil=True, *args, **kwargs):
    if pupil:
      return self.pupil.output(self.path.output(self.source.output(), *args, **kwargs))    
    return self.path.output(self.source.output(), *args, **kwargs)
  