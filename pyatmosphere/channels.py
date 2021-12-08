from matplotlib import pyplot as plt

from pyatmosphere.theory.atmosphere import get_rytov2
from pyatmosphere.measures import I
from pyatmosphere.utils import CrossRef


class Channel:
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
