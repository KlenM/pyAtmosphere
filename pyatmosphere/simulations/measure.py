import numpy as np
from typing import Sequence


class Measure:
    def __init__(
        self,
        channel,
        measure_type: str,
        *operations,
        name: str = "",
        max_size: int = None,
        time: Sequence[float] = None,
        save_path: str = None,
        save_name: str = None
    ):
        self.channel = channel
        self.measure_type: str = measure_type
        self.operations: tuple = tuple(operations)
        self.name = name or (operations[0].__name__ if len(
            operations) == 1 and operations[0].__name__ != "<lambda>" else "")
        self.max_size = max_size
        self.time: tuple = tuple(time) if time else None
        self.data: list = []
        self.iteration_data = None

    @property
    def is_done(self):
        return self.max_size is not None and len(self) >= self.max_size

    def __len__(self):
        return len(self.data)

    def __array__(self):
        return np.asarray(self.data)

    def __repr__(self):
        return self.name

#   def __eq__(self, other):
#     is_channels_equal = self.channel == other.channel
#     is_measure_types_equal = self.measure_type == other.measure_type
#     is_operations_equal = self.operations == other.operations
#     return is_channels_equal and is_measure_types_equal and is_operations_equal

#   def __hash__(self):
#     return hash((self.channel, self.measure_type, self.operations))