def I(channel, output=None, *args, **kwargs):
    if not output is None:
        return abs(output)**2
    return abs(channel.run(*args, **kwargs))**2


def eta(channel, *args, **kwargs):
    return (I(channel, *args, **kwargs).sum(axis=(-1, -2)) * channel.grid.delta**2).item()


def mean_x(channel, *args, **kwargs):
    kwargs["pupil"] = False
    return ((I(channel, *args, **kwargs) * channel.grid.get_x()).sum(axis=(-1, -2)) * channel.grid.delta**2).item()


def mean_y(channel, *args, **kwargs):
    kwargs["pupil"] = False
    return ((I(channel, *args, **kwargs) * (-1) * channel.grid.get_y()).sum(axis=(-1, -2)) * channel.grid.delta**2).item()

# def mean_r(channel, *args, **kwargs):
#   kwargs["pupil"] = False
#   intensity = I(channel, *args, **kwargs)
#   return (np.sqrt((intensity * channel.grid.get_x()).sum(axis=(-1, -2))**2 + (intensity * channel.grid.get_y()).sum(axis=(-1, -2))**2) * channel.grid.delta**2).item()


def mean_x2(channel, *args, **kwargs):
    kwargs["pupil"] = False
    return ((I(channel, *args, **kwargs) * channel.grid.get_x()**2).sum(axis=(-1, -2)) * channel.grid.delta**2).item()


def mean_xy(channel, *args, **kwargs):
    kwargs["pupil"] = False
    return ((I(channel, *args, **kwargs) * (-1 * channel.grid.get_x() * channel.grid.get_y())).sum(axis=(-1, -2)) * channel.grid.delta**2).item()


def mean_y2(channel, *args, **kwargs):
    kwargs["pupil"] = False
    return ((I(channel, *args, **kwargs) * channel.grid.get_y()**2).sum(axis=(-1, -2)) * channel.grid.delta**2).item()

# def mean_r2(channel, *args, **kwargs):
#   kwargs["pupil"] = False
#   return ((I(channel, *args, **kwargs) * channel.grid.get_rho2()).sum(axis=(-1, -2)) * channel.grid.delta**2).item()
