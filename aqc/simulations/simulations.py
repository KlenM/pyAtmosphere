class Simulations:
    def __init__(self, channel, simulations: list):
        self.channel = channel
        self.simulations = simulations

    def run(self):
        while True:
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