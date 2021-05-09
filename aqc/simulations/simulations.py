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
                if simulation.type == "phase_screen":
                    simulation.process(phase_screen, propagation_step)
                    simulation.iteration += 1
                elif simulation.type == "propagation":
                    simulation.process(propagation_result, propagation_step)
        for simulation in self.simulations:
            if simulation.type == "output":
                simulation.process(self.channel.output)
            if simulation.type == "pupil":
                simulation.process(self.channel.pupil.output(self.channel.output))

            if simulation.type != "phase_screen":
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
                self.simulations[active_simulation].print()
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
