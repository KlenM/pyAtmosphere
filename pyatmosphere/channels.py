from matplotlib import pyplot as plt
import numpy as np

from pyatmosphere.theory.atmosphere import get_rytov2
from pyatmosphere.measures import I
from pyatmosphere.utils import CrossRef
from pyatmosphere.gpu import get_array

from .grids import RectGrid, RandLogPolarGrid
from .sources import GaussianSource
from .pathes import IdenticalPhaseScreensPath
from .phase_screens import SSPhaseScreen
from .theory.models import MVKModel
from .pupils import CirclePupil


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
        plt.imshow(get_array(I(self, *args, **kwargs)), extent=self.grid.extent)


def QuickChannel(
        Cn2=1e-15,
        length=1e3,
        count_ps=5,
        beam_w0=0.09,
        beam_wvl=808e-9,
        aperture_radius=0.02,
        grid_resolution=1024,
        grid_delta=0.001
        ):
    quick_channel = Channel(
        grid=RectGrid(resolution=grid_resolution, delta=grid_delta),
        source=GaussianSource(wvl=beam_wvl, w0=beam_w0, F0=np.inf),
        path=IdenticalPhaseScreensPath(
            phase_screen=SSPhaseScreen(
                model=MVKModel(Cn2=Cn2, l0=3e-3, L0=1e3),
                f_grid=RandLogPolarGrid(
                    points=2**10,
                    f_min=1 / 1e3 / 15,
                    f_max=1 / 3e-3 * 2
                    )
                ),
            length=length,
            count=count_ps
            ),
        pupil=CirclePupil(radius=aperture_radius)
        )
    return quick_channel
