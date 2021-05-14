from numpy import isin
from aqc.simulations.simulation import OutputSimulation, PhaseScreenSimulation, PropagationSimulation, PupilSimulation
from IPython import display
import ipywidgets as widgets
from jupyter_ui_poll import ui_events


class Simulations:
    def __init__(self, channel, simulations: list):
        self.channel = channel
        self.simulations = simulations

    def iter(self):
        for propagation_step, (propagation_result, phase_screen) in enumerate(self.channel.generator(pupil=False, store_output=True)):
            for simulation in self.simulations:
                if isinstance(simulation, PhaseScreenSimulation):
                    simulation.process_phase_screen(phase_screen)
                    simulation.iteration += 1
                elif isinstance(simulation, PropagationSimulation):
                    simulation.process_propagation(propagation_result, propagation_step)
        for simulation in self.simulations:
            if isinstance(simulation, PropagationSimulation):
                simulation.process_propagation(self.channel.output, len(self.channel.path.phase_screens))
            elif isinstance(simulation, OutputSimulation):
                simulation.process_output(self.channel.output)
            elif isinstance(simulation, PupilSimulation):
                simulation.process_pupil(self.channel.pupil.output(self.channel.output))

            if not isinstance(simulation, PhaseScreenSimulation):
                simulation.iteration += 1

    def run(self):
        while True:
            self.iter()


class SimulationsGUI(Simulations):
    def __init__(self, plot_skip: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_skip = plot_skip
        self.iteration = 0
        self.simulations_dropdown = widgets.Dropdown(
            options=[(type(simulation).__name__, i)for i, simulation in enumerate(self.simulations)],
            value=0,
            description='Simulation:',
            )

    def update_display(self):
        if self.iteration % self.plot_skip == 0:
            display.display(self.simulations_dropdown)
            active_simulation = self.simulations_dropdown.value
            if self.simulations[active_simulation].iteration > 0:
                self.simulations[active_simulation].plot_output()
            print(f"Iteration: {self.iteration}")
            display.clear_output(wait=True)

    def run(self):
        self.update_display()
        with ui_events() as poll:
            while True:
                self.iter()
                self.iteration += 1
                poll(1)
                self.update_display()
