import numpy as np
from matplotlib import pyplot as plt

from pyatmosphere.theory.phase_screens.sf import calculate_sf
from pyatmosphere.gpu import get_array

from .simulation import Measures
from .result import Result


def calculate_structure_function(channel, output):
    return get_array(calculate_sf(output).mean(axis=1))


class StructureFunctionResult(Result):
    def __init__(self, channel, *args, **kwargs):
        measures = [Measures(channel, "phase_screen",
                             calculate_structure_function)]
        super().__init__(*args, channel=channel, measures=measures, **kwargs)
        self.init_theoretical()

    def init_theoretical(self):
        channel = self.measures[0].channel
        k = 2 * np.pi / channel.path.phase_screen.wvl
        self.get_theoretical = channel.path.phase_screen.model.sf_phi(
            self.r, k, channel.path.phase_screen.thickness)
        self.get_numerical_theoretical = channel.path.phase_screen.model.sf_phi_numeric(
            self.r, k, channel.path.phase_screen.thickness)

    @property
    def structure_function(self):
        return get_array(np.asarray(self.measures[0]).mean(axis=0))

    @property
    def r(self):
        return np.arange(1, self.channel.grid.resolution[0]) * self.channel.grid.delta

    def plot_output(self):
        plt.plot(self.r, self.structure_function, label="Simulated")
        plt.plot(self.r, self.get_theoretical, label="Theoretical")
        plt.plot(self.r, self.get_numerical_theoretical,
                 label="Theoretical numerical")
        plt.plot(np.nan, np.nan,
                 label=f"Iterations: {len(self.measures[0])}", alpha=0)
        plt.legend()
        plt.show()
