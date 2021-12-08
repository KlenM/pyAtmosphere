import numpy as np
from matplotlib import pyplot as plt

from aqc.theory.atmosphere import get_rytov2
from aqc.measures import I


class CrossRef:
    def __init__(self, cross_ref_name):
        self.cross_ref_name = cross_ref_name

    def __set_name__(self, obj, name):
        self.privat_name = "_" + name

    def __get__(self, obj, type=None):
        return getattr(obj, self.privat_name)

    def __set__(self, obj, value):
        if not value:
            return
        setattr(obj, self.privat_name, value)
        setattr(value, self.cross_ref_name, obj)


class AQC:
    grid = CrossRef("channel")
    source = CrossRef("channel")
    path = CrossRef("channel")
    pupil = CrossRef("channel")

    def __init__(self, grid, source, path, pupil=None, name=""):
        self.grid = grid
        self.source = source
        self.path = path
        self.pupil = pupil
        self.output = None
        self.name = name

    def run(self, pupil=True, *args, **kwargs):
        if pupil:
            return self.pupil.output(self.path.output(self.source.output(), *args, **kwargs))
        else:
            return self.path.output(self.source.output(), *args, **kwargs)

    def generator(self, pupil=True, store_output=True, *args, **kwargs):
        self.output = None
        if store_output:
            path_output = yield from self.path.generator(self.source.output(), *args, **kwargs)
            self.output = self.pupil.output(
                path_output) if pupil else path_output
        else:
            yield from self.path.generator(self.source.output(), *args, **kwargs)

    def get_rythov2(self):
        return get_rytov2(self.path.phase_screen.model.Cn2, self.source.k, self.path.length)

    def plot(self, *args, **kwargs):
        plt.imshow(I(self, *args, **kwargs).get(), extent=self.grid.extent)
