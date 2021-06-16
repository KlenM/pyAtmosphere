import numpy as np
from scipy.integrate import dblquad
from scipy.special import i0, hyp1f1, hyp2f1

from aqc.theory.atmosphere import get_rytov2, get_r0s


def get_SI_andrews_weak_kolmogorov(length, model, gaussian_beam):
  """
  Laser Beam Propagation through Random Media, 2nd Edition
  Larry C. Andrews, Ronald L. Phillips
  /10.1117/3.626196 (p.274: 40,41)
  """
  if np.isfinite(gaussian_beam.F0):
    raise ValueError("The function has been implemented only for collimated beam.")
  
  SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
  r0_s = get_r0s(model.Cn2, length, gaussian_beam.k)
  c_r = 2 * np.pi
  r_cw2_mean = 2.42 * model.Cn2 * length**3 / gaussian_beam.w0**(1/3) * hyp2f1(1/3, 1, 4, length / gaussian_beam.F0)
  Spe = r_cw2_mean * (1 - (c_r**2 * gaussian_beam.w0**2 / r0_s)/(1 + c_r**2 * gaussian_beam.w0**2 / r0_s))**(1 / 6)
  return 3.86 * SR2 * np.real(1j**(5 / 6) * hyp2f1(-5/6, 11/6, 17/6, 1 - gaussian_beam.get_theta(length) + 1j * gaussian_beam.get_Lambda(length))) - 2.64 * SR2 * gaussian_beam.get_Lambda(length)**(5/6) * hyp1f1(-5/6, 1, 2 * Spe**2 / gaussian_beam.get_w(length)**2)


def get_SI_andrews_strong_asymptotic_zeroscale(length, model, gaussian_beam):
  """
  Laser Beam Propagation through Random Media, 2nd Edition
  Larry C. Andrews, Ronald L. Phillips
  /10.1117/3.626196 (p.332: 23)
  """
  SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
  if np.any(SR2  < 1):
    raise ValueError("Condition \Sigma_R^2 >> 1 is not fulfilled")
  return 1 + (0.86 + 1.87 * (1 - gaussian_beam.get_theta(length))) / SR2**(2 / 5)


def get_SI_andrews_strong_asymptotic_l0(length, model, gaussian_beam):
  """
  Laser Beam Propagation through Random Media, 2nd Edition
  Larry C. Andrews, Ronald L. Phillips
  /10.1117/3.626196 (p.263: 24)
  """
  SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
  Ql = 10.89 * length / gaussian_beam.k / model.l0**2
  if np.any(SR2 * Ql**(7 / 6) < 100):
    raise ValueError("Condition \Sigma_R^2 Q_l >> 100 is not fulfilled")
  return 1 + (2.39 + 5.26 * (1 - gaussian_beam.get_theta(length))) / (SR2 * Ql**(7 / 6))**(1 / 6)


def get_SI_andrews_strong_zeroscale(length, model, gaussian_beam, debug=False):
  """
  Laser Beam Propagation through Random Media, 2nd Edition
  Larry C. Andrews, Ronald L. Phillips
  /10.1117/3.626196 (p.352: 102)
  """
  SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
  SB2 = 3.86 * SR2 * np.real(1j**(5/6) * hyp2f1(-5/6, 11/6, 17/6, (1 - gaussian_beam.get_theta(length)) + 1j * gaussian_beam.get_Lambda(length)) - 11 / 16 * gaussian_beam.get_Lambda(length)**(5/6))

  ### !!! ###
  ### SB**(12 / 5) == SB2**(6 / 5)
  ### !!! ###
  SlnX2 = 0.49 * SB2 / (1 + 0.56 * (1 + gaussian_beam.get_theta(length)) * SB2**(6 / 5))**(7 / 6)
  SlnY2 = 0.51 * SB2 / (1 + 0.69 * SB2**(6 / 5))**(5 / 6)
  return np.exp(SlnX2 + SlnY2) - 1

def get_SI_chan_zhang(length, model, gaussian_beam):
    def SI2(SR2, A, theta, Spe2, W):
        under_re = 1j**(5/6) * hyp2f1(-5/6, 11/6, 17/6, 1 - theta + 1j * A)
        return 3.86 * SR2 * np.real(under_re) - 2.64 * SR2 * A**(5/6) * hyp1f1(-5/6, 1, 2 * Spe2 / W**2)

    def Spe2(rce2_m, W0, r0sp2):
        Cr = 1.5 * np.pi
        return rce2_m * (1 - ((Cr**2 * W0**2 / r0sp2) / (1 + Cr**2 * W0**2 / r0sp2))**(1/6))

    def r0sp2(k, Cn2, L):
        return (0.16 * k**2 * Cn2 * L)**(-3/5)

    def rce2_m(Cn2, L, W0, F0):
        return 2.42 * Cn2 * L**3 * W0**(-1/3) * hyp2f1(1/3, 1, 4, L/F0)
    
    rce2_m_val = rce2_m(model.Cn2, length, gaussian_beam.w0, gaussian_beam.F0)
    r0sp2_val = r0sp2(gaussian_beam.k, model.Cn2, length)
    Spe2_val = Spe2(rce2_m_val, gaussian_beam.w0, r0sp2_val)
    SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
    return SI2(SR2, gaussian_beam.get_Lambda(length), gaussian_beam.get_theta(length), Spe2_val, gaussian_beam.get_w(length))
    

def get_SI_andrews_strong(length, model, gaussian_beam, debug=False):
  """
  Laser Beam Propagation through Random Media, 2nd Edition
  Larry C. Andrews, Ronald L. Phillips
  /10.1117/3.626196 (p.356: 114)
  """
  Ql = 10.89 * length / gaussian_beam.k / model.l0**2
  Q0 = 64 * np.pi**2 * length / gaussian_beam.k / model.L0**2
  SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
  phi_1 = np.arctan(2 * gaussian_beam.get_Lambda(length) / (1 + 2 * gaussian_beam.get_theta(length)))
  phi_2 = np.arctan((1 + 2 * gaussian_beam.get_theta(length)) * Ql / (3 + 2 * gaussian_beam.get_Lambda(length) * Ql))
  
  SG2_1 = 0.4 * ((1 + 2 * gaussian_beam.get_theta(length))**2 + (2 * gaussian_beam.get_Lambda(length) + 3 / Ql)**2)**(11 / 12) / ((1 + 2 * gaussian_beam.get_theta(length))**2 + 4 * gaussian_beam.get_Lambda(length)**2)**(1 / 2)
  SG2_2 = np.sin(11 / 6 * phi_2 + phi_1)
  SG2_3 = 2.61 / ((1 + 2 * gaussian_beam.get_theta(length))**2 * Ql**2 + (3 + 2 * gaussian_beam.get_Lambda(length) * Ql)**2)**(1 / 4) * np.sin(4 / 3 * phi_2 + phi_1)
  SG2_4 = 0.52 / ((1 + 2 * gaussian_beam.get_theta(length))**2 * Ql**2 + (3 + 2 * gaussian_beam.get_Lambda(length) * Ql)**2)**(7 / 24) * np.sin(5 / 4 * phi_2 + phi_1)
  SG2_5 = 13.40 * gaussian_beam.get_Lambda(length) / Ql**(11 / 6) / ((1 + 2 * gaussian_beam.get_theta(length))**2 + 4 * gaussian_beam.get_Lambda(length)**2)
  SG2_6 = 11 / 6 * ((1 + 0.31 * gaussian_beam.get_Lambda(length) * Ql)**(5 / 6) + (1.1 * (1 + 0.27 * gaussian_beam.get_Lambda(length) * Ql)**(1 / 3)) - (0.19 * (1 + 0.24 * gaussian_beam.get_Lambda(length) * Ql)**(1 / 4))) / Ql**(5 / 6)
  SG2 = 3.86 * SR2 * (SG2_1 * (SG2_2 + SG2_3 - SG2_4) - SG2_5 - SG2_6)

  eta_X_1 = 0.38 / (1 - 3.21 * (1 - gaussian_beam.get_theta(length)) + 5.29 * (1 - gaussian_beam.get_theta(length))**2)
  eta_X_2 = 0.47 * SR2 * Ql**(1 / 6) * ((1 / 3 - 1 / 2 * (1 - gaussian_beam.get_theta(length)) + 1 / 5 * (1 - gaussian_beam.get_theta(length))**2) / (1 + 2.2 * (1 - gaussian_beam.get_theta(length))))**(6 / 7)
  eta_X = (eta_X_1 + eta_X_2)**(-1)
  eta_X0 = eta_X * Q0 / (eta_X + Q0)
 
  SlnX2l0_1 = 1 / 3 - 1 / 2 * (1 - gaussian_beam.get_theta(length)) + 1 / 5 * (1 - gaussian_beam.get_theta(length))**2
  SlnX2l0_2 = (eta_X * Ql / (eta_X + Ql))**(7 / 6)
  SlnX2l0_3 = 1.75 * (eta_X / (eta_X + Ql))**(1 / 2)
  SlnX2l0_4 = 0.25 * (eta_X / (eta_X + Ql))**(7 / 12)
  SlnX2l0 = 0.49 * SR2 * SlnX2l0_1 * SlnX2l0_2 * (1 + SlnX2l0_3 - SlnX2l0_4)
  
  SlnX2L0_1 = 1 / 3 - 1 / 2 * (1 - gaussian_beam.get_theta(length)) + 1 / 5 * (1 - gaussian_beam.get_theta(length))**2
  SlnX2L0_2 = (eta_X0 * Ql / (eta_X0 + Ql))**(7 / 6)
  SlnX2L0_3 = 1.75 * (eta_X0 / (eta_X0 + Ql))**(1 / 2)
  SlnX2L0_4 = 0.25 * (eta_X0 / (eta_X0 + Ql))**(7 / 12)
  SlnX2L0 = 0.49 * SR2 * SlnX2L0_1 * SlnX2L0_2 * (1 + SlnX2L0_3 - SlnX2L0_4)

  SlnY2l0 = 0.51 * SG2 / (1 + 0.69 * SG2**(6 / 5))**(5 / 6)

  return np.exp(SlnX2l0 - SlnX2L0 + SlnY2l0) - 1


# def get_SI_andrews_weak(r, length, model, gaussian_beam):
#   """
#   !!! Not verified !!!

#   Optical scintillations and fade statistics for a satellite-communication system
#   L. C. Andrews, R. L. Phillips, and P. T. Yu
#   /10.1364/AO.34.007742 (20)

#   Laser Beam Propagation through Random Media, 2nd Edition
#   Larry C. Andrews, Ronald L. Phillips
#   /10.1117/3.626196 (p.263: 14)
#   """

#   def inintegral(z, kappa):
#     return kappa * model.psd_n(kappa) * np.exp(-gaussian_beam.get_Lambda(length) * length * kappa**2 / gaussian_beam.k * ((length - z) / length)**2) * (i0(2 * gaussian_beam.get_Lambda(length) * r * kappa * (length - z) / length) - np.cos(length * kappa**2 / gaussian_beam.k * ((length - z) / length) * (gaussian_beam.get_theta(length) - (1 - gaussian_beam.get_theta(length)) * z / length)))
#   return 8 * np.pi**2 * gaussian_beam.k**2 * dblquad(inintegral, 0, length, 0, np.inf)[0]


# def get_SI_andrews_weak_kolmogorov(r, length, model, gaussian_beam):
#   """
#   !!! Not verified !!!

#   Laser Beam Propagation through Random Media, 2nd Edition
#   Larry C. Andrews, Ronald L. Phillips
#   /10.1117/3.626196 (p.263: 18,19)
#   """
#   SR2 = get_rytov2(model.Cn2, gaussian_beam.k, length)
#   return 3.86 * SR2 * np.real(1j**(5 / 6) * hyp2f1(-5/6, 11/6, 17/6, 1 - gaussian_beam.get_theta(length) + 1j * gaussian_beam.get_Lambda(length))) - 2.64 * SR2 * gaussian_beam.get_Lambda(length)**(5/6) * hyp1f1(-5/6, 1, 2 * r**2 / gaussian_beam.get_w(length)**2)

