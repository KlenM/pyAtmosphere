import numpy as np
from aqc.theory.atmosphere import get_r0s
from aqc.theory.atmosphere.gamma2 import get_gamma_2


# def get_w_LT(length, model, gaussian_beam):
#   """
#   Scintillation and beam-wander analysis in an optical ground station–satellite uplink
#   Federico Dios, Juan Antonio Rubio, Alejandro Rodrı´guez, and Adolfo Comero´ n
#   /10.1364/AO.43.003866 (2)
#   """
#   vacuum_part = gaussian_beam.get_w(length)
#   turbulent_part = np.sqrt(2) * 4 * length / gaussian_beam.k / get_r0s(model.Cn2, length, gaussian_beam.k)
#   return np.sqrt(vacuum_part**2 + turbulent_part**2)

def get_numeric_w_LT(L, model, w0, wvl, F, rho, delta):
  gamma_2 = np.array([get_gamma_2(i, L, model, w0, wvl, F) for i in rho])
  gamma_2 = gamma_2 / ((gamma_2 * rho).sum() * delta)
  return np.sqrt(2 * (gamma_2 * rho**3).sum() * delta)
