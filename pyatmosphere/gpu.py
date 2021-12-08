import numpy


config = {
    "use_gpu": False,
}


def get_xp():
    if config['use_gpu']:
        import cupy
        return cupy
    else:
        return numpy


def get_array(xp_array):
    if config['use_gpu']:
        return xp_array.get()
    return xp_array
