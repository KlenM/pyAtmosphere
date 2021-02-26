import numpy as np
import scipy.integrate


def get_r_bw(L, model, gaussian_beam):
  """Source ???"""
  def drc2(z, kappa, L):    
    def Dsp(rho):
      kappa0 = (2 * np.pi) / model.L0
      return 1.09 * model.Cn2 *  gaussian_beam.k**2 * L * model.l0**(-1/3) * rho**2 * (1/(1+(rho/model.l0)**2)**(1/6) - 0.72 * (kappa0 * model.l0)**(1/3))
    return 4 * np.pi**2 * (L - z)**2 * kappa**3 * model.psd_n(kappa) * np.exp(-(kappa * gaussian_beam.get_w(z))**2 / 4 - np.pi * Dsp(kappa * z / gaussian_beam.k))
  
  return np.sqrt(scipy.integrate.dblquad(drc2, 0, np.inf, 0, L, args=(L, ))[0])

# def xBW2_weak(Cn2, k, w, L): 
#   """Source ???"""
#   return 0.33 * w**2 * get_rytov2(Cn2, k, L) * get_fresnel(k, w, L)**(-7/6)
