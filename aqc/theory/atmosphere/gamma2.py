import numpy as np
import scipy.integrate
import scipy.special


def get_gamma_2(r, L, model, w0, wvl, F, epsabs=1e-12, limit=300):
    """
    Precision analysis of turbulence phase screens and their influence on the simulation of Gaussian beam propagation in turbulent atmosphere
    Zhibin Chen, Dongxiao Zhang, Cheng Xiao, and Mengze Qin
    https://doi.org/10.1364/AO.389121 (24)
    """

    k = 2 * np.pi / wvl
    w = w0 * np.sqrt((1 - L/F)**2 + (2 * L / k / w0**2)**2)
    A = 2 * L / k / w**2
    kappa0 = (2 * np.pi) / model.L0

    def gamma2_prima(Q):
        def Dsp_vk(rho):
            return 1.09 * model.Cn2 * k**2 * L * model.l0**(-1/3) * rho**2 * (1/(1+(rho/model.l0)**2)**(1/6) - 0.72 * (kappa0 * model.l0)**(1/3))
        return Q * scipy.special.jv(0, k * r * Q / L) * np.exp(- k * Q**2 / 4 / A / L - 1 / 2 * Dsp_vk(Q))

    return (k * w0 / 2 / L)**2 * scipy.integrate.quad(gamma2_prima, 0, np.inf, epsabs=epsabs, limit=limit)[0]
