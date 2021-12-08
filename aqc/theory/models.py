import cupy
import numpy as np
import scipy.integrate
import scipy.special
from dataclasses import dataclass

from aqc.theory.atmosphere import get_r0


@dataclass
class Model:
    Cn2: float
    l0: float
    L0: float

    def psd_n_f(self, f):
        return self.psd_n(2 * np.pi * f)

    def psd_phi(self, kappa, k, thickness):
        return 2 * np.pi * k**2 * thickness * self.psd_n(kappa)

    def psd_phi_f(self, f, k, thickness):
        return 2 * np.pi * k**2 * thickness * self.psd_n_f(f)

    def sf_phi_numeric(self, r, k, thickness):
        xp = cupy.get_array_module(r)

        def dsf(f, r):
            return self.psd_n(2 * np.pi * f) * (1 - scipy.special.jn(0, (2 * np.pi * f * r).item())) * 2 * np.pi * f

        phi_coeff = 2 * xp.pi * k**2 * thickness
        return xp.array([phi_coeff * (2 * xp.pi) * 2 * scipy.integrate.quad(dsf, 0, np.inf, args=(ri,), epsrel=1e-3,)[0] * (2*xp.pi) for ri in r])


class KModel(Model):
    def psd_n(self, kappa):
        cupy_psd_n = cupy.ElementwiseKernel(
            'float32 kappa, float32 Cn2, float32 l0, float32 L0',
            'float32 Fn',
            '''if ( kappa < 1./L0 ) {
        Fn = 0;
      } else if ( kappa < 1./l0 ) {
        Fn = 0.033 * Cn2 * powf(kappa, -11./3);
      } else {
        Fn = 0;
      }'''
        )
        return cupy_psd_n(kappa, self.Cn2, self.l0, self.L0)

    def sf(self, r):
        cupy_sf = cp.ElementwiseKernel(
            'float32 r, float32 Cn2, float32 l0, float32 L0',
            'float32 Dn',
            '''if ( r < l0 ) {
        Dn = Cn2 * powf(l0, -4./3) * powf(r, 2);
      } else if ( r < L0 ) {
        Dn = Cn2 * powf(r, 2./3);
      } else {
        Dn = 0;
      }'''
        )
        return cupy_sf(r, self.Cn2, self.l0, self.L0)


class TModel(Model):
    def psd_n(self, kappa):
        cupy_psd_n = cupy.ElementwiseKernel(
            'float32 k, float32 Cn2, float32 l0, float32 L0',
            'float32 Fn',
            '''if ( k < 1./L0 ) {
        Fn = 0;
      } else {
        Fn = 0.033 * Cn2 * exp(-powf(k / (5.92 / l0), 2)) * powf(k, -11./3);
      }''',
            'tatarski_psd_n'
        )
        return cupy_psd_n(kappa, self.Cn2, self.l0, self.L0)


class MVKModel(Model):
    def psd_n(self, kappa):
        xp = cupy.get_array_module(kappa)
        k0 = (2 * xp.pi) / self.L0
        km = 5.92 / self.l0
        return 0.033 * self.Cn2 * xp.exp(-(kappa / km)**2) / (kappa**2 + k0**2)**(11/6)

    def sf_phi(self, r, k, thickness):
        xp = cupy.get_array_module(r)
        k0 = (2 * xp.pi) / self.L0
        r0 = get_r0(self.Cn2, k, thickness)
        return 7.75 * r0**(-5/3) * self.l0**(-1/3) * r**2 * (1/(1 + 2.03 * r**2 / self.l0**2)**(1/6) - 0.72 * (k0 * self.l0)**(1/3))


class AndrewsModel(Model):
    def psd_n(self, kappa):
        xp = cupy.get_array_module(kappa)
        kl = 3.3 / l0
        k0 = (2 * np.pi) / L0
        k_per_kl = k / kl
        return 0.033 * Cn2 * (1 + 1.802 * k_per_kl - 0.254 * k_per_kl**(7/6)) * \
            xp.exp(-(k_per_kl)**2) / (k**2 + k0**2)**(11/6)
