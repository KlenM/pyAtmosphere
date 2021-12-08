import numpy as np
import pandas as pd
from typing import Sequence
from matplotlib import pyplot as plt
from aqc.simulation import Measures
from aqc.results import Result
from aqc.measures import eta, mean_x, mean_y
from scipy.stats import pearsonr


class WindResult(Result):
    def load_output(self):
        for measures, data in zip(self.measures, pd.read_csv(self.save_path).T.values):
            measures.data = [
                [float(i) for i in row[1:-1].split(", ")] for row in data]


class TimeCoherenceResult(WindResult):
    def __init__(self, channel, time, *args, **kwargs):
        measures = [Measures(channel, "pupil", eta, time=time)]
        super().__init__(*args, channel=channel, measures=measures, **kwargs)

    @property
    def tc(self) -> Sequence[float]:
        return [pearsonr(np.asarray(self.measures[0])[:, 0], np.asarray(self.measures[0])[:, i])[0] for i in range(len(self.measures[0].time))]

    def plot_output(self):
        if len(self.measures[0]) > 2:
            plt.plot(self.measures[0].time, self.tc)
            plt.ylim((0, 1))
        plt.show()
        print(f"Iteration: {len(self.measures[0].data)}")


class TimeBWcorrSimulation(WindResult):
    def __init__(self, channel, time, *args, **kwargs):
        measures = [Measures(channel, "atmosphere", mean_x, time=time), Measures(
            channel, "atmosphere", mean_y, time=time)]
        super().__init__(*args, channel=channel, measures=measures, **kwargs)

    @property
    def xx(self) -> Sequence[float]:
        return 2 * np.sqrt((np.asarray(self.measures[0])[:, 0, None] * np.asarray(self.measures[0])[:, :]).mean(axis=0))

    @property
    def yy(self) -> Sequence[float]:
        return 2 * np.sqrt((np.asarray(self.measures[1])[:, 0, None] * np.asarray(self.measures[1])[:, :]).mean(axis=0))

    @property
    def xy(self) -> Sequence[float]:
        return 2 * np.sqrt(abs((np.asarray(self.measures[0])[:, 0, None] * np.asarray(self.measures[1])[:, :]).mean(axis=0)))

    def plot_output(self):
        plt.scatter(self.measures[0].time, self.xx)
        plt.ylabel(
            "Bean wandering $2 \cdot \\sqrt{{\\left<x_0 x_{{\tau}}\\right>}}$, m")
        plt.xlabel("Wind shift, m")
        plt.show()

        plt.scatter(self.measures[0].time, self.xy)
        plt.ylabel(
            f"Bean wandering $2 \cdot \\sqrt{{|\\left<x_0 y_{{\tau}}\\right>|}}$, m")
        plt.xlabel("Wind shift, m")
        plt.show()
