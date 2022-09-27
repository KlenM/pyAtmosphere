from typing import Optional, Tuple

import numpy as np
from scipy.special import lambertw, iv


class EllipticBeamAnalyticalPDT:
    def __init__(self, W0, a, size: int):
        self.W0 = W0
        self.a = a
        self.size = size
        self.bw: Optional[float] = None
        self.theta_mean: Optional[float] = None
        self.theta_cov: Optional[Tuple[float, float]] = None

    def set_params(self, bw: float, theta_mean: float, theta_cov: Tuple[float, float]):
        self.bw = bw
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        return self

    def set_params_from_data(self, mean_x, mean_x2, mean_y2):
        bw = np.sqrt((mean_x**2).mean())
        W2_mean, W2_cov = self._get_mean_W(mean_x, mean_x2, mean_y2)
        theta_mean, theta_cov = self._get_mean_theta(W2_mean, W2_cov)
        return self.set_params(bw, theta_mean, theta_cov)

    def _get_mean_W(self, mean_x, mean_x2, mean_y2):
        x2_mean = mean_x2.mean()
        xx_mean = (mean_x**2).mean()
        x2x2_mean = (mean_x2**2).mean()
        x2y2_mean = (mean_x2 * mean_y2).mean()
        W2_mean = 4 * (x2_mean - xx_mean)

        delta_ij = np.asarray([1, 0])
        part_1 = 8 * delta_ij * xx_mean**2
        part_2 = xx_mean * W2_mean
        part_3 = x2x2_mean * (4 * delta_ij - 1) - x2y2_mean * (4 * delta_ij - 3)
        Wi2Wj2_mean = 8 * (-part_1 - part_2 + part_3)
        W2_cov = Wi2Wj2_mean - W2_mean**2
        return W2_mean, W2_cov

    def _get_mean_theta(self, W2_mean, W2_cov):
        theta_mean = np.log(W2_mean / self.W0**2 / np.sqrt(1 + W2_cov[0] / W2_mean**2))
        theta_cov = np.log(1 + W2_cov / W2_mean**2)
        return theta_mean, theta_cov

    def _get_W_eff(self, chi, W1, W2):
        exp_part1 = self.a**2 / W1**2 * (1 + 2 * np.cos(chi)**2)
        exp_part2 = self.a**2 / W2**2 * (1 + 2 * np.sin(chi)**2)
        arg = 4 * self.a**2 / W1 / W2 * np.exp(exp_part1 + exp_part2)
        W2_eff = 4 * self.a**2 / lambertw(arg).real
        return np.sqrt(W2_eff)

    def _get_R_lambda(self, xi) -> Tuple[float, float]:
        if xi == 0:
            return np.inf, 2
        arg = self.a**2 * xi**2
        exp_bes_part = 1 - np.exp(-arg) * iv(0, arg)
        log_part = np.log(2 * (1 - np.exp(-arg / 2)) / exp_bes_part)

        lmbd_part_1 = np.exp(-arg) * iv(1, arg)
        lmbd = 2 * arg * lmbd_part_1 / exp_bes_part / log_part
        R = log_part**(-1 / lmbd)
        return R, lmbd

    def _get_eta0(self, W1, W2):
        R, lmbd = self._get_R_lambda(1 / W1 - 1 / W2)
        eta0_part1 = iv(0, self.a**2 * (1 / W1**2 - 1 / W2**2))
        eta0_part2 = np.exp(-self.a**2 * (1 / W1**2 + 1 / W2**2))
        if W1 == W2:
            eta0_part3 = 0
            eta0_part4 = 0
        else:
            eta0_part3 = 2 * (1 - np.exp(-self.a**2 / 2 * (1 / W1 - 1 / W2)**2))
            eta0_part4 = np.exp(-((W1 + W2)**2 / np.abs(W1**2 - W2**2) / R)**lmbd)
        return 1 - eta0_part1 * eta0_part2 - eta0_part3 * eta0_part4

    def eta(self, r_0, varphi_0, theta_1, theta_2, phi):
        W1 = self.W0 * np.exp(theta_1 / 2)
        W2 = self.W0 * np.exp(theta_2 / 2)

        W_eff = self._get_W_eff(phi - varphi_0, W1=W1, W2=W2)
        eta_0 = self._get_eta0(W1=W1, W2=W2)
        R, lmbd = self._get_R_lambda(2 / W_eff)
        return eta_0 * np.exp(-(r_0 / self.a / R)**lmbd)

    def pdt(self):
        if self.bw is None or self.theta_mean is None or self.theta_cov is None:
            raise ValueError(
                'The parametes must be setted via .set_params(...) or .set_params_from_data(...).')
        r_0s = np.random.rayleigh(self.bw, size=self.size)
        varphi_0s = np.random.uniform(0, 2 * np.pi, size=self.size)
        thetas = np.random.multivariate_normal(
            [self.theta_mean, self.theta_mean],
            [self.theta_cov, self.theta_cov[::-1]],
            size=self.size).T
        phis = np.random.uniform(0, np.pi / 2, size=self.size)
        params = zip(r_0s, varphi_0s, thetas[0], thetas[1], phis)
        transmittance = []
        for r_0, varphi_0, theta_1, theta_2, phi in params:
            transmittance.append(self.eta(r_0, varphi_0, theta_1, theta_2, phi))
        return transmittance
