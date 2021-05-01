import numpy as np
import cupy


config = {
  "use_gpu": True,
  "dtype": {
    "float": np.float32,
    "complex": np.complex64
  }
}


class CrossRef:
  def __init__(self, cross_ref_name):
    self.cross_ref_name = cross_ref_name
    
  def __set_name__(self, obj, name):
    self.privat_name = "_" + name
    
  def __get__(self, obj, type=None):
    return getattr(obj, self.privat_name)
  
  def __set__(self, obj, value):
    if not value:
      return
    setattr(obj, self.privat_name, value)
    setattr(value, self.cross_ref_name, obj)


class AQC:
  grid = CrossRef("channel")
  source = CrossRef("channel")
  path = CrossRef("channel")
  pupil = CrossRef("channel")
  
  def __init__(self, grid, source, path, pupil=None):
    self.grid = grid
    self.source = source
    self.path = path
    self.pupil = pupil

  def run(self, pupil=True, *args, **kwargs):
    if pupil:
      return self.pupil.output(self.path.output(self.source.output(), *args, **kwargs))    
    return self.path.output(self.source.output(), *args, **kwargs)
  