from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

from aqc.grid import RectGrid
from aqc.utils import ifft2


@dataclass
class PhaseScreen():
  model: object
  grid: object = None
  wvl: float = None
  thickness: float = None

  def generate_phase_screen(self):
    """Return complex phase screen"""
    raise NotImplementedError

  def generate(self, complex=False):
    if complex:
      return self.generate_phase_screen()
    return self.generate_phase_screen().real
  
  def generator(self):
    while True:
      ps = self.generate(complex=True)
      yield ps.real
      yield ps.imag
  

@dataclass
class FFTPhaseScreen(PhaseScreen):
  subharmonics: int = 4

  def generate_phase_screen(self):
    xp = self.grid.get_array_module()
    def get_cn_coefficients(cn_f_grid):
      cn = (xp.random.normal(size=cn_f_grid.shape) + 1j * xp.random.normal(size=cn_f_grid.shape)) * \
            xp.sqrt(self.model.psd_phi_f(cn_f_grid.get_rho(), 2 * xp.pi / self.wvl, self.thickness)) * 2 * xp.pi * cn_f_grid.delta
      cn[cn_f_grid.origin_index] = 0
      return cn
    
    f_grid = self.grid.get_f_grid()
    phase_screen = ifft2(get_cn_coefficients(f_grid), 1)

    for sh in range(self.subharmonics):
      sh_f_grid = RectGrid(3, f_grid.delta / 3**(sh + 1), xp=xp)
      cn = get_cn_coefficients(sh_f_grid)

      # fx, fy = sh_f_grid.get_xy()
      # return xp.exp(1j * 2 * xp.pi * self.grid.get_y() @ fy.T) @ cn @ xp.exp(1j * 2 * xp.pi * fx.T @ self.grid.get_x())
      f = sh_f_grid.get_x()
      for i in range(sh_f_grid.resolution):
        for j in range(sh_f_grid.resolution):
          phase_screen = phase_screen + cn[i,j] * xp.exp(1j * 2 * xp.pi * (f[0, i] * self.grid.get_x() + f[0, j] * self.grid.get_y()))
    
    return phase_screen - xp.mean(phase_screen)
  

@dataclass
class SSPhaseScreen(PhaseScreen):
  f_grid: object = None

  def __post_init__(self):
    self._sqrt_int_spectrum = None
  
  @property
  def sqrt_int_spectrum(self):
    if not self._sqrt_int_spectrum is None:
      return self._sqrt_int_spectrum
    
    xp = self.grid.get_array_module()
    f = self.f_grid.base
    self._sqrt_int_spectrum = xp.empty(self.f_grid.points)
    in_int_function = lambda f: (2 * np.pi)**2 * f * self.model.psd_phi_f(f, 2 * xp.pi / self.wvl, self.thickness)

    for i in range(self.f_grid.points):
      f_prev = f[i - 1] if i != 0 else 0
      self._sqrt_int_spectrum[i] = xp.sqrt(2 * np.pi * quad(in_int_function, f_prev, f[i])[0])
    
    return self._sqrt_int_spectrum

  def generate_phase_screen(self):
    xp = self.grid.get_array_module()
    
    cn = xp.array([1, 1j]) @ xp.random.normal(size=(2, self.f_grid.points)) * self.sqrt_int_spectrum

    rho = self.f_grid.get_rho()
    theta = self.f_grid.get_theta()
    fx, fy = self.f_grid.get_xy(rho, theta)
    return xp.exp(1j * 2 * xp.pi * self.grid.get_y() @ fy.T) @ xp.diag(cn) @ xp.exp(1j * 2 * xp.pi * fx.T @ self.grid.get_x())

  

@dataclass
class SUPhaseScreen(PhaseScreen):
  f_grid: object = None

  def __post_init__(self):
    self._delta_k_base = None
  
  @property
  def delta_k_base(self):
    xp = self.grid.get_array_module()
    if self._delta_k_base is None:
      self._delta_k_base = (2 * xp.pi)**2 * xp.array((self.f_grid.base**2 - np.insert(self.f_grid.base, 0, 0)[:-1]**2))
    return self._delta_k_base

  def generate_phase_screen(self):
    xp = self.grid.get_array_module()

    rho = self.f_grid.get_rho()
    theta = self.f_grid.get_theta()
    
    cn = xp.array([1, 1j]) @ xp.random.normal(size=(2, self.f_grid.points)) * \
      xp.sqrt(self.model.psd_phi_f(rho, 2 * xp.pi / self.wvl, self.thickness) * xp.pi * self.delta_k_base) 

    fx, fy = self.f_grid.get_xy(rho, theta)
    return xp.exp(1j * 2 * xp.pi * self.grid.get_y() @ fy.T) @ xp.diag(cn) @ xp.exp(1j * 2 * xp.pi * fx.T @ self.grid.get_x())


@dataclass
class WindSUPhaseScreen(PhaseScreen):
  speed: float = 10
  grid: object = None
  f_grid: object = None

  def __post_init__(self):
    self.cnp = None

  def generate_cn(self):
    self.rho = self.f_grid.get_rho()
    self.theta = self.f_grid.get_theta()
    xp = self.grid.get_array_module()
    self.cnp = xp.array([1, 1j]) @ xp.random.normal(size=(2, self.f_grid.points))
    self.iteration = 0

  def generate_phase_screen(self):
    if self.cnp is None:
      self.generate_cn()

    xp = self.grid.get_array_module()

    cn = self.cnp * \
      xp.sqrt(self.model.psd_phi_f(self.rho, 2 * xp.pi / self.wvl, self.thickness) * \
      xp.pi * (2 * xp.pi)**2 * xp.array((self.f_grid.base**2 - np.insert(self.f_grid.base, 0, 0)[:-1]**2))) 
    
    fx, fy = self.f_grid.get_xy(self.rho, self.theta)
    offset = self.iteration * self.speed
    self.iteration += 1
    return xp.exp(1j * 2 * xp.pi * self.grid.get_y() @ fy.T) @ xp.diag(cn) @ xp.exp(1j * 2 * xp.pi * fx.T @ (self.grid.get_x() + offset))

  def generator(self):
    while True:
      yield self.generate(complex=False)
  