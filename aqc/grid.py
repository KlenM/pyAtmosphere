from dataclasses import dataclass

import numpy as np
import cupy

from aqc.aqc import AQCElement, config


@dataclass
class Grid(AQCElement):
  def get_array_module(self):
    return cupy if config["use_gpu"] else np


@dataclass
class RectGrid(Grid):
  resolution: tuple
  delta: float
  dtype: object = config["dtype"]["float"]
  
  def __post_init__(self):
    if isinstance(self.resolution, int):
      self.resolution = (self.resolution, self.resolution)
    else:
      self.resolution = self.resolution

  @property
  def size(self):
    return np.array(self.resolution) * self.delta
  
  @property
  def shape(self):
    return self.resolution

  @property
  def origin_index(self):
    return (self.resolution[0] // 2, self.resolution[1] // 2)
  
  @property
  def _left_bound(self):
    return -self.resolution[0] // 2 + bool(self.resolution[0] % 2)
  
  @property
  def _right_bound(self):
    return self.resolution[0] // 2 + bool(self.resolution[0] % 2)
  
  @property
  def _top_bound(self):
    return -self.resolution[1] // 2 + bool(self.resolution[1] % 2)
  
  @property
  def _bottom_bound(self):
    return self.resolution[1] // 2 + bool(self.resolution[1] % 2)
  
  def get_NxNy(self):
    xp = self.get_array_module()
    return xp.ogrid[self._top_bound:self._bottom_bound, self._left_bound:self._right_bound]

  def get_N2(self):
    Nx, Ny = self.get_NxNy()
    return Nx**2 + Ny**2

  def get_x(self):
    xp = self.get_array_module()
    return xp.arange(self._left_bound, self._right_bound, dtype=self.dtype).reshape((1, -1)) * self.delta

  def get_y(self):
    xp = self.get_array_module()
    return xp.arange(self._top_bound, self._bottom_bound, dtype=self.dtype).reshape((-1,1)) * self.delta
  
  def get_xy(self):
    return self.get_x(), self.get_y()

  def get_rho2(self):
    y, x = self.get_xy()
    return x**2 + y**2
  
  def get_rho(self):
    xp = self.get_array_module()
    return xp.sqrt(self.get_rho2())
  
  def get_f_grid(self):
    f_grid = RectGrid(resolution=int(np.min(self.resolution)), delta=1 / (np.min(self.resolution) * self.delta))
    return f_grid



@dataclass
class RandLogPolarGrid(Grid):
  points: int
  f_min: float
  f_max: float
  dtype: object = config["dtype"]["float"]


  @property
  def base(self):
    return np.exp(np.linspace(np.log(self.f_min), np.log(self.f_max), self.points, dtype=self.dtype))

  def get_rho(self):
    xp = self.get_array_module()
    rand = np.random.random()
    f = self.base
    f_prev = np.insert(f, 0, 0)[:-1]
    return xp.array(np.sqrt(f_prev**2 + rand * (f**2 - f_prev**2)))

  def get_theta(self):
    xp = self.get_array_module()
    return xp.random.random(size=(self.points,)) * 2 * xp.pi

  def get_x(self, rho, theta):
    xp = self.get_array_module()
    return rho * xp.cos(theta)

  def get_y(self, rho, theta):
    xp = self.get_array_module()
    return rho * xp.sin(theta)
  
  def get_xy(self, rho, theta):
    xp = self.get_array_module()
    return (rho * xp.cos(theta)).reshape((1, -1)), (rho * xp.sin(theta)).reshape((-1, 1))

  def plot(self):
    lim = self.f_min * 10
    lim = self.f_max

    plt.figure(figsize=(8,8))
    plt.xlim((-lim,lim))
    plt.ylim((-lim,lim))
    c = plt.Circle((0,0), self.f_min, fill=False)
    plt.gca().add_patch(c)
    c = plt.Circle((0,0), self.f_max, fill=False)
    plt.gca().add_patch(c)
    for i in range(3):
      q = grid.get_xy()
      plt.scatter(q[0].get(), q[1].T.get(), s=8)
  