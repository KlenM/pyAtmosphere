from aqc.gpu import get_xp


def calculate_sf(input):
    xp = get_xp()
    a = [xp.sum((input[:, :-r] - input[:, r:])**2, axis=1) /
        (input.shape[1] - r) for r in range(1, input.shape[1])]
    return xp.array(a)
