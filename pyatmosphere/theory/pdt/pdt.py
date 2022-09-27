import numpy as np
import warnings

from scipy.stats import lognorm, beta
from scipy.special import i1
from scipy.integrate import quad

from pyatmosphere.measures import eta as eta_measure
from pyatmosphere.pupils import CirclePupil
from pyatmosphere.grids import RectGrid
from pyatmosphere.gpu import get_xp


def bw_eta_0(a, st2):
    return 1 - np.exp(-2 * a**2 / st2)


def bw_shape_l(eta_0, a, st2):
    return 8 * a**2 / st2 * (np.exp(-4 * a**2 / st2) * i1(4 * a**2 / st2) / (1 - np.exp(-4 * a**2 / st2) * np.i0(4 * a**2 / st2))) * np.log(2 * eta_0 / (1 - np.exp(-4 * a**2 / st2) * np.i0(4 * a**2 / st2)))**(-1)


def bw_scale_R(eta_0, a, l, st2):
    return a * np.log(2 * eta_0 / (1 - np.exp(-4 * a**2 / st2) * np.i0(4 * a**2 / st2)))**(-1 / l)


def bw_pdt(eta, eta_0, R, l, bw2):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message=r"divide by zero encountered in", category=RuntimeWarning)
        warnings.filterwarnings(
            'ignore', message=r"invalid value encountered in (power|multiply)", category=RuntimeWarning)
        pdt = R**2 / (bw2 * eta * l) * np.log(eta_0 / eta)**(2 / l - 1) * \
            np.exp(-(R**2 / (2 * bw2)) * np.log(eta_0 / eta)**(2 / l))
        pdt[np.isnan(pdt)] = 0
        return pdt


def lognormal_pdt(eta, eta_mean, eta2_mean):
    mu = -np.log(eta_mean**2 / np.sqrt(eta2_mean))
    sigma = np.sqrt(np.log(eta2_mean / eta_mean**2))
    lognorm_model = lognorm(sigma, scale=np.exp(-mu))
    return lognorm_model.pdf(eta) / lognorm_model.cdf(1)


def beam_wandering_pdt(eta, a, st2, bw2, eta_0=None, shape_l=None, scale_R=None):
    eta_0 = eta_0 or bw_eta_0(a, st2)
    shape_l = shape_l or bw_shape_l(eta_0, a, st2)
    scale_R = scale_R or bw_scale_R(eta_0, a, shape_l, st2)
    return bw_pdt(eta, eta_0, scale_R, shape_l, bw2)


def beta_pdt(eta, eta_mean, eta2_mean):
    eta_std2 = eta2_mean - eta_mean**2
    beta_a = (eta_mean**2 - eta_mean**3 - eta_mean * eta_std2) / eta_std2
    beta_b = beta_a * (1 / eta_mean - 1)
    return beta.pdf(eta, beta_a, beta_b)


def bayesian_pdt(eta, eta_mean, eta2_mean, a, st2, bw2, scale_R=None, shape_l=None, r0_size=2000):
    bw = np.sqrt(bw2)

    def get_eta0(eta_mean, bw, scale_R, shape_l):
        def under_int(xi):
            return xi * np.exp(-xi**2 / 2) * np.exp(-(bw / scale_R * xi)**shape_l)
        return eta_mean / quad(under_int, 0, np.inf)[0]

    def get_zeta02(eta2_mean, bw, scale_R, shape_l):
        def under_int(xi):
            return xi * np.exp(-xi**2 / 2) * np.exp(-2 * (bw / scale_R * xi)**shape_l)
        return eta2_mean / quad(under_int, 0, np.inf)[0]

    eta_0 = bw_eta_0(a, st2)
    shape_l = shape_l or bw_shape_l(eta_0, a, st2)
    scale_R = scale_R or bw_scale_R(eta_0, a, shape_l, st2)
    eta0 = get_eta0(eta_mean, bw, scale_R, shape_l)
    zeta02 = get_zeta02(eta2_mean, bw, scale_R, shape_l)
    sigma_r0 = np.sqrt(np.log(zeta02 / eta0**2))

    total_probability_model = np.zeros_like(eta)
    for r0 in np.random.rayleigh(bw, size=r0_size):
        mu_r0 = -np.log(eta0**2 / np.sqrt(zeta02)) + \
            (abs(r0) / scale_R)**shape_l
        lognorm_model = lognorm(sigma_r0, scale=np.exp(-mu_r0))
        total_probability_model += lognorm_model.pdf(
            eta) / lognorm_model.cdf(1)
    return 1 / r0_size * total_probability_model


def beta_bayesian_pdt(eta, eta_mean, eta2_mean, a, st2, bw2, scale_R=None, shape_l=None, r0_size=2000):
    bw = np.sqrt(bw2)

    def get_eta0(eta_mean, bw, scale_R, shape_l):
        def under_int(xi):
            return xi * np.exp(-xi**2 / 2) * np.exp(-(bw / scale_R * xi)**shape_l)
        return eta_mean / quad(under_int, 0, np.inf)[0]

    def get_zeta02(eta2_mean, bw, scale_R, shape_l):
        def under_int(xi):
            return xi * np.exp(-xi**2 / 2) * np.exp(-2 * (bw / scale_R * xi)**shape_l)
        return eta2_mean / quad(under_int, 0, np.inf)[0]

    eta_0 = bw_eta_0(a, st2)
    shape_l = shape_l or bw_shape_l(eta_0, a, st2)
    scale_R = scale_R or bw_scale_R(eta_0, a, shape_l, st2)
    eta0 = get_eta0(eta_mean, bw, scale_R, shape_l)
    zeta02 = get_zeta02(eta2_mean, bw, scale_R, shape_l)

    beta_bayesian_model = np.zeros_like(eta)
    for r0 in np.random.rayleigh(bw, size=r0_size):
        eta_mean_r0 = eta0 * np.exp(-(r0 / scale_R)**shape_l)
        eta2_mean_r0 = zeta02 * np.exp(-2 * (r0 / scale_R)**shape_l)
        beta_bayesian_model += beta_pdt(eta, eta_mean_r0, eta2_mean_r0)
    return 1 / r0_size * beta_bayesian_model


def elliptic_beam_numerical_transmission(beam_params: dict, pupil_radiuses, resolution=2**8, is_tracked=False):
    mean_x, mean_y = beam_params["mean_x"], beam_params["mean_y"]
    mean_x2, mean_y2, mean_xy = beam_params["mean_x2"], beam_params["mean_y2"], beam_params["mean_xy"]

    lt_mean = np.sqrt(4 * (np.asarray(mean_x2) + np.asarray(mean_y2)).mean())

    class DummyChannel:
        grid = RectGrid(resolution=resolution, delta=8 * lt_mean / resolution)

    pupils = []
    for pupil_radius in pupil_radiuses:
        pupil = CirclePupil(radius=pupil_radius)
        pupil.channel = DummyChannel
        pupils.append(pupil)

    model_eta = [[] for _ in pupils]
    for mean_x, mean_y, mean_x2, mean_y2, mean_xy in zip(mean_x, mean_y, mean_x2, mean_y2, mean_xy):
        Sxx = 4 * (mean_x2 - mean_x**2)
        Syy = 4 * (mean_y2 - mean_y**2)
        Sxy = 4 * (mean_xy - mean_x * mean_y)
        detS = Sxx * Syy - Sxy**2
        xmx0 = DummyChannel.grid.get_x()[0] - mean_x
        ymy0 = -1 * DummyChannel.grid.get_y().T[0] - mean_y
        xp = get_xp()
        X, Y = xp.meshgrid(xmx0, ymy0)
        intensity = np.sqrt(2 / np.pi / np.sqrt(detS)) * \
            xp.exp(-(Syy * X**2 - 2 * Sxy * X * Y + Sxx * Y**2) / detS)
        for i, pupil in enumerate(pupils):
            shift = (0, 0) if not is_tracked else (mean_x, mean_y)
            model_eta[i].append(eta_measure(
                DummyChannel, output=intensity * pupil.get_pupil(shift=shift)))
    return model_eta
