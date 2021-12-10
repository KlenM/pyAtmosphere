# Atmospheric-Quantum-Channels

Physics-based simulation of light propagation in turbulent atmosphere.

## Installation
*Highly recommend to use [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages.

```bash
pip install -r requirements.txt
```

If you want to use GPU for simulation, you need to [install](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) `CUDA` on your machine.
Additinally, the `cupy` python package is required. For example, for CUDA 11.0:
```bash
pip install cupy-cuda110
```

## Usage
### GPU usage

If you want to enable GPU simulation, execute at the beginning of your script:

```python
from pyatmosphere import gpu
gpu.config['use_gpu'] = True
```

### QuickChannel example

```python
from pyatmosphere import QuickChannel

quick_channel = QuickChannel(
    Cn2=1e-15,
    length=10000,
    count_ps=5,
    beam_w0=0.09,
    beam_wvl=8.08e-07,
    aperture_radius=0.12
    )

quick_channel.plot()
```

### Advanced channel

```python
import numpy as np
from pyatmosphere import (
    Channel,
    RectGrid,
    RandLogPolarGrid,
    GaussianSource,
    IdenticalPhaseScreensPath,
    SSPhaseScreen,
    CirclePupil,
    MVKModel,
    measures
    )

channel = Channel(
    grid=RectGrid(
        resolution=2048, 
        delta=0.0015
    ),
    source=GaussianSource(
        wvl=808e-9,
        w0=0.12,
        F0=np.inf
    ),
    path=IdenticalPhaseScreensPath(
        phase_screen=SSPhaseScreen(
            model=MVKModel(
                Cn2=5e-16,
                l0=6e-3,
                L0=1e3,
            ),
            f_grid=RandLogPolarGrid(
                points=2**10, 
                f_min=1 / 1e3 / 15, 
                f_max=1 / 6e-3 * 2
            )
        ),
        length=50e3,
        count=5
    ),
    pupil=CirclePupil(
        radius=0.2
    ),
)

channel_output = channel.run(pupil=False)
intensity = measures.I(channel, output=channel_output)
mean_x = measures.mean_x(channel, output=channel_output)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Feel free to open new issues if you have any questions.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
