import numpy as np
from typing import Tuple, Sequence
from matplotlib import pyplot as plt

from pyatmosphere.measures import I, mean_x, mean_y, mean_x2, mean_xy, mean_y2  # , mean_r, mean_r2
from pyatmosphere.theory.atmosphere.beam_wandering import get_r_bw
from pyatmosphere.theory.atmosphere.long_term import get_numeric_w_LT
from pyatmosphere.gpu import get_array

from .measure import Measure
from .result import Result


class BeamResult(Result):
    def __init__(self, channel, **kwargs):
        measures = [
            Measure(channel, "atmosphere", mean_x),
            Measure(channel, "atmosphere", mean_y),
            Measure(channel, "atmosphere", mean_x2),
            Measure(channel, "atmosphere", mean_xy),
            Measure(channel, "atmosphere", mean_y2),
            Measure(channel, "atmosphere", self.mean_x2_r, name="mean_x2_r"),
        ]
        super().__init__(channel, measures, **kwargs)

    def mean_x2_r(self, channel, output):
        r0 = np.sqrt(self.measures[0].iteration_data **
                     2 + self.measures[1].iteration_data**2)
        cosXi = self.measures[0].iteration_data / r0
        sinXi = self.measures[1].iteration_data / r0
        rotated_x2_r = (channel.grid.get_x() * cosXi +
                        ((-1) * channel.grid.get_y()) * sinXi)**2
        return ((I(channel, output=output) * rotated_x2_r).sum(axis=(-1, -2)) * channel.grid.delta**2).item()

    @property
    def bw2(self) -> Sequence[float]:
        x2 = np.asarray(self.measures[0])**2
        return x2

    @property
    def lt2(self) -> Sequence[float]:
        r2 = 4 * np.asarray(self.measures[2])
        return r2

    @property
    def st2(self) -> Sequence[float]:
        return self.lt2 - 4 * self.bw2

    @property
    def bw(self) -> Tuple[float, float]:
        bw2 = self.bw2
        bw_mean = np.sqrt(bw2.mean())
        Dbw2 = bw2.std(ddof=1)/np.sqrt(len(bw2))
        Dbw = Dbw2 / 2 / bw_mean
        return bw_mean, Dbw

    @property
    def lt(self) -> Tuple[float, float]:
        lt2 = self.lt2
        lt_mean = np.sqrt(lt2.mean())
        Dlt2 = lt2.std(ddof=1)/np.sqrt(len(lt2))
        Dlt = Dlt2 / 2 / lt_mean
        return lt_mean, Dlt

    @property
    def st(self) -> Tuple[float, float]:
        st2 = self.st2
        st_mean = np.sqrt(st2.mean())
        Dst2 = st2.std(ddof=1)/np.sqrt(len(st2))
        Dst = Dst2 / 2 / st_mean
        return st_mean, Dst

    def print_output(self):
        bw_result = self.bw
        lt_result = self.lt
        st_result = self.st
        print(f"sigma_BW_x = {bw_result[0]:.1e} +- {bw_result[1]:.1e}")
        print(f"sigma_LT_x = {lt_result[0]:.1e} +- {lt_result[1]:.1e}")
        print(f"W_ST = {st_result[0]:.1e} +- {st_result[1]:.1e}")
        print(f"Count of measures: {len(self.measures[0])}")


class BeamPropagationResult(Result):
    def __init__(self, channel, **kwargs):
        measures = [Measure(channel, "propagation", mean_r),
                    Measure(channel, "propagation", mean_r2)]
        super().__init__(channel, measures, **kwargs)

        self.bw_theoretical = [get_r_bw(
            L, self.channel.path.phase_screen.model, self.channel.source) for L in self.positions]
        rho = get_array(self.channel.grid.get_x()[
                           0, self.channel.grid.origin_index[0]::4])

        def w_LT(L): return get_numeric_w_LT(L, self.channel.path.phase_screen.model, self.channel.source.w0,
                                             self.channel.source.wvl, self.channel.source.F0, rho, self.channel.grid.delta)
        self.lt_theoretical = [w_LT(L) for L in self.positions]

    @property
    def bw2(self) -> Sequence[float]:
        r2 = np.asarray(self.measures[0])**2
        return r2

    @property
    def lt2(self) -> Sequence[float]:
        r2 = 2 * np.asarray(self.measures[1])
        return r2

    @property
    def st2(self) -> Sequence[float]:
        return self.lt2 - self.bw2

    @property
    def positions(self):
        return np.array(list(self.channel.path.positions) + [self.channel.path.length])

    def plot_output(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 9))
        axs[0].set_ylabel(r"Beam wandering $\left<r_c\right>$, m")
        axs[0].plot(self.positions, np.sqrt(
            self.bw2.mean(axis=0)), label="Simulated")
        axs[0].plot(self.positions, self.bw_theoretical, label="Theory")
        axs[0].plot(np.nan, np.nan,
                    label=f"Iterations: {len(self.measures[0])}", alpha=0)
        axs[0].legend()

        axs[1].set_ylabel(r"Long term $W_{LT}$, m")
        axs[1].plot(self.positions, np.sqrt(
            self.lt2.mean(axis=0)), label="Simulated")
        axs[1].plot(self.positions, self.lt_theoretical, label="Theory")
        axs[1].plot(np.nan, np.nan,
                    label=f"Iterations: {len(self.measures[0])}", alpha=0)
        axs[1].legend()

        axs[2].set_ylabel(r"Short term $W_{ST}$, m")
        axs[2].plot(self.positions, np.sqrt(self.st2.mean(axis=0)))
        axs[2].set_xlabel("Propagation distance z, m")
        axs[2].plot(np.nan, np.nan,
                    label=f"Iterations: {len(self.measures[0])}", alpha=0)
        axs[2].legend()
        plt.show()
