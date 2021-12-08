import numpy as np
from matplotlib import pyplot as plt

from pyatmosphere.simulation import Measures
from pyatmosphere.results import Result
from pyatmosphere.theory.atmosphere.si import get_SI_andrews_strong
from pyatmosphere.gpu import get_array


def intensity_at_center(channel, output):
    return get_array(abs(output[channel.grid.origin_index[0], channel.grid.origin_index[1]])**2)


class SIResult(Result):
    def __init__(self, channel, theoretical_functions=(get_SI_andrews_strong,), *args, **kwargs):
        measures = [Measures(channel, "propagation", intensity_at_center)]
        super().__init__(*args, channel=channel, measures=measures, **kwargs)
        self.set_theoretical_functions(*theoretical_functions)

    def set_theoretical_functions(self, *theoretical_functions):
        self.theoretical_functions = theoretical_functions
        self.theoretical_si = [theoretical_function(
            self.positions, self.channel.path.phase_screen.model, self.channel.source) for theoretical_function in theoretical_functions]

    @property
    def intensities_at_center(self):
        return np.asarray(self.measures[0])

    @property
    def positions(self):
        return np.array(list(self.channel.path.positions) + [self.channel.path.length])

    @property
    def si(self):
        return (self.intensities_at_center**2).mean(axis=0) / self.intensities_at_center.mean(axis=0)**2 - 1

    def plot_output(self):
        plt.plot(self.positions, self.si,
                 label=r"On-axis SI $\sigma_I$, m", **self.plot_kwargs)
        for i, theoretical_function in enumerate(self.theoretical_functions):
            plt.plot(
                self.positions, self.theoretical_si[i], label=f"Theoretical on-axis SI: {theoretical_function.__name__}")
        plt.plot(np.nan, np.nan,
                 label=f"Iterations: {len(self.measures[0])}", alpha=0)
        plt.xlabel("Propagation distance z, m")
        plt.legend()
        plt.show()
