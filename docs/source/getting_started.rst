Getting Started
===============

Installation
------------
Highly recommend to use `virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_.

Use the package manager `pip <https://pip.pypa.io/en/stable/>`_ to install required packages:

.. code-block:: console

    pip install -r requirements.txt

If you want to use GPU acceleration for simulation, you need to `install <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation>`_ ``CUDA`` on your machine.
Additinally, the `cupy` python package is required. For example, for CUDA 11.0:

.. code-block:: console

    pip install cupy-cuda110

Then, you have to :ref:`Turn on GPU acceleration`.

Quick channel
-------------

.. code-block::

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

For further dipping read :ref:`User Guide`.

