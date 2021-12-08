import numpy as np
from matplotlib import pyplot as plt
from functools import partial, lru_cache

from pyatmosphere.measures import eta, mean_x, mean_y

from .simulation import Measures
from .result import Result


class PDTResult(Result):
    def __init__(self, channel, pupils: list = None, **kwargs):
        self.pupil_shift = (0, 0)
        self.pupils = pupils or [channel.pupil]
        measures = kwargs.pop("measures", [
            Measures(channel, "atmosphere", partial(
                self.append_pupil, pupil), eta, name=f"{pupil.radius}")
            for pupil in self.pupils])
        super().__init__(channel, measures, **kwargs)

    def append_pupil(self, pupil, channel, output):
        init_pupil = channel.pupil
        channel.pupil = pupil
        output = channel.pupil.output(output, shift=self.pupil_shift)
        channel.pupil = init_pupil
        return output

    def plot_output(self):
        if len(self.pupils) == 1:
            plt.hist(
                self.measures[0].data, label=f"Count: {len(self.measures[0])}", bins=200, range=(0, 1))
            plt.legend()
            plt.show()
        else:
            n_x = 3
            n_y = len(self.pupils) // n_x + bool(len(self.pupils) % n_x)
            fig, axes = plt.subplots(n_y, n_x, figsize=(15, 3 * n_y))
            for i, ax in enumerate(axes.flat):
                if i >= len(self.pupils):
                    break
                ax.hist(
                    self.measures[i].data,
                    label=f"Pupil radius: {self.pupils[i].radius:.3f}\nCount: {len(self.measures[0])}",
                    bins=100,
                    range=(0, 1)
                )
                ax.legend()
            plt.show()


class TrackedPDTResult(PDTResult):
    def __init__(self, channel, pupils: list = None, **kwargs):
        pupils = pupils or [channel.pupil]
        beam_measures = [Measures(channel, "atmosphere", mean_x), Measures(
            channel, "atmosphere", mean_y)]
        pdt_measures = [
            Measures(channel, "atmosphere", self.set_pupil_position, partial(
                self.append_pupil, pupil), eta, name=f"{pupil.radius}")
            for pupil in pupils]
        super().__init__(channel, pupils=pupils,
                         measures=beam_measures + pdt_measures, **kwargs)

    def set_pupil_position(self, channel, output):
        self.pupil_shift = (
            self.measures[0].iteration_data, self.measures[1].iteration_data)
        return output
