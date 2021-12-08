import numpy as np
from IPython import display
from pathlib import Path
from uuid import uuid4
from typing import TypedDict, Sequence
from aqc.pupils import CirclePupil
from aqc.results import Result


class Measures:
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


class Simulation:
    def __init__(self, results_list: Sequence[Result] = None, measures_list: Sequence[Measures] = None):
        self.measures = {}
        if measures_list:
            for measures in measures_list:
                self.add_measures(measures)
        self.results_list = results_list
        if results_list:
            for result in results_list:
                for measures in result.measures:
                    self.add_measures(measures)

    def add_measures(self, measures):
        """channel - time - measure_type - operations"""
        if measures.channel not in self.measures:
            self.measures[measures.channel] = {}
        if measures.time not in self.measures[measures.channel]:
            self.measures[measures.channel][measures.time] = {}
        if measures.measure_type not in self.measures[measures.channel][measures.time]:
            self.measures[measures.channel][measures.time][measures.measure_type] = {}
        if measures.operations not in self.measures[measures.channel][measures.time][measures.measure_type]:
            self.measures[measures.channel][measures.time][measures.measure_type][measures.operations] = []
        self.measures[measures.channel][measures.time][measures.measure_type][measures.operations].append(
            measures)

    def init_measures_iteration_data(self):
        for measures in self.flattened_measures():
            empty_data = None
            if measures.time:
                if measures.measure_type == "propagation":
                    empty_data = [[None for _ in range(
                        len(measures.channel.path.positions) + 1)] for _ in measures.time]
                else:
                    empty_data = [None for _ in measures.time]
            elif measures.measure_type == "propagation":
                empty_data = [None for _ in range(
                    len(measures.channel.path.positions) + 1)]
            measures.iteration_data = empty_data

#     self.iter_data = {}
#     for channel, channel_measures in self.measures.items():
#       self.iter_data[channel] = {}
#       for time, time_measures in channel_measures.items():
#         self.iter_data[channel][time] = {}
#         for measures_type, measures_type_measures in time_measures.items():
#           self.iter_data[channel][time][measures_type] = {}
#           for operations, _ in measures_type_measures.items():
#             empty_data = None
#             if time:
#               if measures_type == "propagation":
#                 empty_data = [[None for _ in range(len(channel.path.positions) + 1)] for _ in time]
#               else:
#                 empty_data = [None for _ in time]
#             elif measures_type == "propagation":
#               empty_data = [None for _ in range(len(channel.path.positions) + 1)]
#             self.iter_data[channel][time][measures_type][operations] = empty_data

    def process_operations(self, output, operations_measures, time_id, propagation_id=None):
        for operations, measures_list in operations_measures.items():
            if self.is_measures_done(measures_list):
                continue
            measures = measures_list[0]
            measures_output = output.copy()
            for operation in operations:
                measures_output = operation(
                    measures.channel, output=measures_output)

            for measures in measures_list:
                if measures.is_done:
                    continue

                if not measures.time is None:
                    if not propagation_id is None:
                        measures.iteration_data[time_id][propagation_id] = measures_output
                    else:
                        measures.iteration_data[time_id] = measures_output
                else:
                    if not propagation_id is None:
                        measures.iteration_data[propagation_id] = measures_output
                    else:
                        measures.iteration_data = measures_output

    def iter(self):
        self.init_measures_iteration_data()
        for channel, channel_measures in self.measures.items():
            for ps in channel.path.phase_screens:
                ps.cache_clear()
            for time, time_measures in channel_measures.items():
                time = time or [None]
                for time_id, time_value in enumerate(time):
                    for propagation_id, (propagation_result, phase_screen) in enumerate(channel.generator(pupil=False, shift=(0, time_value or 0), store_output=True, wind=True)):
                        self.process_operations(propagation_result, time_measures.get(
                            "propagation", {}), time_id, propagation_id)
                        if propagation_id == 0:
                            self.process_operations(
                                phase_screen, time_measures.get("phase_screen", {}), time_id)
                    self.process_operations(
                        channel.output, time_measures.get("atmosphere", {}), time_id)
                    self.process_operations(channel.output, time_measures.get(
                        "propagation", {}), time_id, -1)
                    if channel.pupil:
                        self.process_operations(channel.pupil.output(
                            channel.output), time_measures.get("pupil", {}), time_id)
        for measures in self.flattened_measures():
            if not measures.is_done:
                measures.data.append(measures.iteration_data)
#       if not measures.is_done:
#         measures.data.append(self.iter_data[measures.channel][measures.time][measures.measure_type][measures.operations])

    def flattened_measures(self, measures=None):
        measures = measures if measures is not None else self.measures
        if isinstance(measures, dict):
            for key, values in measures.items():
                yield from self.flattened_measures(values)
        else:
            yield from measures

    def is_measures_done(self, measures=None):
        return all((m.is_done for m in self.flattened_measures(measures)))

    def run(self, *args, plot_step: int = None, save_step: int = None, **kwargs):
        try:
            iteration = 0
            while not self.is_measures_done():
                self.iter()
                iteration += 1
                self.process_output(
                    iteration, plot_step=plot_step, save_step=save_step)
        except KeyboardInterrupt:
            pass
        finally:
            self.process_output(0, plot_step=plot_step, save_step=save_step)

    def process_output(self, iteration, plot_step, save_step):
        if plot_step and iteration % plot_step == 0:
            for result in self.results_list:
                result.plot_output()
            display.clear_output(wait=True)

        if save_step and iteration % save_step == 0:
            for result in self.results_list:
                result.save_output()
