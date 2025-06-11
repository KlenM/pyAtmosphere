import numpy as np
from scipy.special import iv


EXP_OVERFLOW_THRESHOLD = np.log(np.finfo(np.float64).max)


def bw_eta_0(a, st2):
    return 1 - np.exp(-2 * a**2 / st2)


def bw_shape_l(eta_0, a, st2):
    return 8 * a**2 / st2 * (np.exp(-4 * a**2 / st2) * iv(1, 4 * a**2 / st2) / (1 - np.exp(-4 * a**2 / st2) * np.i0(4 * a**2 / st2))) * np.log(2 * eta_0 / (1 - np.exp(-4 * a**2 / st2) * np.i0(4 * a**2 / st2)))**(-1)


def bw_scale_R(eta_0, a, l, st2):
    return a * np.log(2 * eta_0 / (1 - np.exp(-4 * a**2 / st2) * np.i0(4 * a**2 / st2)))**(-1 / l)


def bw_pdt(eta, eta_0, R, l, bw2):
    eta = np.asarray(eta)
    pdt = np.zeros_like(eta)
    mask = (0 < eta) & (eta < eta_0)
    eta_less_than_eta0 = eta[mask]
    pdt[mask] = R**2 / (bw2 * eta_less_than_eta0 * l) * np.log(eta_0 / eta_less_than_eta0)**(2 / l - 1) * \
        np.exp(-(R**2 / (2 * bw2)) * np.log(eta_0 / eta_less_than_eta0)**(2 / l))
    return pdt


def bw_is_clear_transmittance(a, st2):
    # If the aperture is about 13 times larger than the beam width, this leads to an overflow of the i0 (i1).

    return 4 * a**2 / st2 > EXP_OVERFLOW_THRESHOLD


def beam_wandering_pdt(eta, st2, bw2, aperture_radius, eta_0=None, shape_l=None, scale_R=None):

    # In the case of i0 overflow, approximate with the maximum possible PDT, which is close to a clear channel anyway.
    if bw_is_clear_transmittance(aperture_radius, st2):
        return bw_pdt(eta, eta_0=1, R=aperture_radius * 1.0113920776113101, l=30.454248857822876, bw2=bw2)

    eta_0 = eta_0 or bw_eta_0(aperture_radius, st2)
    shape_l = shape_l or bw_shape_l(eta_0, aperture_radius, st2)
    scale_R = scale_R or bw_scale_R(eta_0, aperture_radius, shape_l, st2)
    return bw_pdt(eta, eta_0, scale_R, shape_l, bw2)


def get_eta_mean(S_BW, W, aperture_radius):
    return 1 - np.exp(-2 * aperture_radius**2 / (4 * S_BW**2 + W**2))


def get_eta2_mean(S_BW, W, aperture_radius):
    def F_1_approx(nu, r, sigma):
        x2 = r**2 / sigma**2
        F0 = 1 - np.exp(-x2 / 2)
        if x2 < EXP_OVERFLOW_THRESHOLD:
            _iv0 = iv(0, x2)
            _log = np.log(2 * F0 / (1 - _iv0 * np.exp(-x2)))
            if _log < 1e-7:
                return 0.0
            mu_1 = (2 * x2 * (1 / (np.exp(x2) / iv(1, x2) - _iv0 / iv(1, x2))) / _log)
            D1_1 = sigma / r * _log**(1 / mu_1)
        else:
            _iv0 = iv(0, EXP_OVERFLOW_THRESHOLD)
            _log = np.log(2 * F0 / (1 - _iv0 * np.exp(-x2)))
            mu_1 = (2 * x2 * (iv(1, EXP_OVERFLOW_THRESHOLD) / (np.exp(EXP_OVERFLOW_THRESHOLD) - _iv0)) / _log)
            D1_1 = sigma / r * _log**(1 / mu_1)
        result = F0 * np.exp(-(D1_1 * nu / sigma)**mu_1)
        return result

    def Q_approx(nu, r, sigma):
        res = 1 - F_1_approx(nu, r, sigma)
        return res

    p = W**2 / 8 / S_BW**2
    d = 2 * aperture_radius / W * np.sqrt(2 * p * (p + 1) / (2 * p**2 + 3 * p + 1))
    b = 1 / (2 * p + 1)
    res = (1 - 2 * np.exp(-2 * aperture_radius**2 / (4 * S_BW**2 + W**2)) + np.exp(-d**2 / 2) *
            (1 - Q_approx(d, d * b, np.sqrt(1 - b**2)) + Q_approx(d * b, d, np.sqrt(1 - b**2))))
    return res
