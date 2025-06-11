import numpy as np
import scipy
from scipy import optimize

from .beam_wandering import beam_wandering_pdt, get_eta_mean, get_eta2_mean


class CircularBeamModel:
    def __init__(self, S_BW, S_mu, S_sigma2, aperture_radius):
        self.S_BW = S_BW
        self.S_mu = S_mu
        self.S_sigma2 = S_sigma2
        self.aperture_radius = aperture_radius

    def __repr__(self):
        return f"<CircularBeamModel 'S_BW': {self.S_BW:.2e}, 'S_mu': {self.S_mu}, 'S_sigma2': {self.S_sigma2:.2e}, 'aperture_radius': {self.aperture_radius:.2e}>"

    @classmethod
    def from_beam_params(cls, S_BW, W2_mean, W4_mean, aperture_radius):
        S_mu = np.log(W2_mean**2 /  np.sqrt(W4_mean))
        S_sigma2 = np.log(W4_mean / W2_mean**2)
        return cls(S_BW, S_mu, S_sigma2, aperture_radius)

    @classmethod
    def from_raw_data(cls, beam_data, aperture_radius):
        S_BW = np.sqrt((beam_data['mean_x']**2).mean())
        W2 = 4 * (beam_data['mean_x2'] - beam_data['mean_x']**2)
        W4 = W2**2
        W2_mean = W2.mean()
        W4_mean = W4.mean()
        return cls.from_beam_params(S_BW, W2_mean, W4_mean, aperture_radius)

    def get_S_distribution(self, range_precision=0.0001):
        S_distribution = scipy.stats.lognorm(s=np.sqrt(self.S_sigma2), scale=np.exp(self.S_mu))
        S_min = S_distribution.ppf(range_precision)
        S_max = S_distribution.ppf(1 - range_precision)
        return S_distribution, (S_min, S_max)

    def get_pdt(self, transmittance, range_precision=0.0001, quad_limit=100):
        S_distribution, (S_min, S_max) = self.get_S_distribution(range_precision)

        def _integrand(S, eta):
            return beam_wandering_pdt(np.asarray([eta]), S, self.S_BW**2, self.aperture_radius) * S_distribution.pdf(S)

        vectorized_integration = np.vectorize(lambda eta: scipy.integrate.quad(_integrand, S_min, S_max, args=(eta,), limit=quad_limit)[0])
        return vectorized_integration(transmittance)

    def get_eta_mean(self):
        S_distribution, (S_min, S_max) = self.get_S_distribution()
        def _integrand(S):
            return get_eta_mean(self.S_BW, np.sqrt(S), self.aperture_radius) * S_distribution.pdf(S)
        return scipy.integrate.quad(_integrand, S_min, S_max)[0]

    def get_eta2_mean(self):
        S_distribution, (S_min, S_max) = self.get_S_distribution()
        def _integrand(S):
            return get_eta2_mean(self.S_BW, np.sqrt(S), self.aperture_radius) * S_distribution.pdf(S)
        return scipy.integrate.quad(_integrand, S_min, S_max, limit=150)[0]


class AnchoredCircularBeamModel(CircularBeamModel):
    @classmethod
    def from_beam_params(cls, S_BW, eta_mean, eta2_mean, aperture_radius, initial_guess_W2_mean, initial_guess_W4_mean):
        def system_of_equations(v, S_BW, aperture_radius, eta_mean, eta2_mean, path=None):
            S_mu = v[0]
            S_sigma2 = v[1]
            cb_model = CircularBeamModel(S_BW=S_BW, S_mu=S_mu, S_sigma2=S_sigma2, aperture_radius=aperture_radius)
            d1 = cb_model.get_eta_mean() - eta_mean
            d2 = cb_model.get_eta2_mean() - eta2_mean
            if path is not None:
                path.append({'S_mu': v[0], 'S_sigma2': v[1], 'd_eta': d1, 'd_eta2': d2})
            return d1**2 + d2**2


        # Assume average S can't be bigger(less) than 5(5^-1) times of initial guess
        mu_bound = (np.log(initial_guess_W2_mean / 5) - 1, np.log(5 * initial_guess_W2_mean))
        # Assume S isn't bigger than 10 times of average S with probability 0.99
        sigma2_bound = (1e-6, 2)

        cb_initial_guess = CircularBeamModel.from_beam_params(S_BW, initial_guess_W2_mean, initial_guess_W4_mean, aperture_radius)
        S_mu, S_sigma2 = scipy.optimize.minimize(system_of_equations, [cb_initial_guess.S_mu, cb_initial_guess.S_sigma2],
            args=(S_BW, aperture_radius, eta_mean, eta2_mean), bounds=[mu_bound, sigma2_bound], method='Nelder-Mead').x

        return cls(S_BW, S_mu, S_sigma2, aperture_radius)

    @classmethod
    def from_raw_data(cls, transmittance_data, beam_data, aperture_radius):
        S_BW = np.sqrt((beam_data['mean_x']**2).mean())
        eta_mean = transmittance_data.mean()
        eta2_mean = (transmittance_data**2).mean()
        W2 = 4 * (beam_data['mean_x2'] - beam_data['mean_x']**2)
        W4 = W2**2
        return cls.from_beam_params(S_BW, eta_mean, eta2_mean, aperture_radius,  W2.mean(), W4.mean())
