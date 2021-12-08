from dataclasses import dataclass


@dataclass
class CirclePupil:
    radius: float

    def get_pupil(self, shift=(0, 0)):
        x, y = self.channel.grid.get_xy()
        return ((x - shift[0])**2 + (y + shift[1])**2 <= (self.radius)**2)

    def output(self, input, **kwargs):
        return input * self.get_pupil(**kwargs)
