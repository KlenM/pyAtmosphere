# import numpy as np
# import cupy as cp
# from matplotlib import pyplot as plt

# from aqc.atmosphere.utils import Cn2_to_r0, psd_k_to_f, psd_n_to_phi
# from aqc.utils import pp


# def subharmonics(simulation, structure_function_phi, alpha_Phi_value=0.99):
#   appropriate_subharmonics_n = appropriate_subharmonics(simulation, alpha_Phi_value)
#   print("Approproate nums of subharmonics: ", pp(appropriate_subharmonics_n))
#   print("For", simulation.atmosphere.phase_screens[-1]["subharmonics"], "subharmonics:")
#   print("alpha_Phi = ", pp(alpha_Phi(simulation)), ", beta_D =", pp(beta_D(simulation, structure_function_phi)))

# def alpha_Phi(simulation):
#   """
#   The integrated power ratio: alpha_Phi = IP_{actual} / IP_{theory}
#   DOI: 10.1080/09500340.2019.1596323
#   """
#   phase_screen = simulation.atmosphere.phase_screens[-1]
#   flim = 0.5 / (3 ** phase_screen["subharmonics"] * simulation.grid.size)
#   f0 = 1 / phase_screen["L0"]
#   return (1 + (flim / f0)**2)**(-5/6)

# def beta_D(simulation, structure_function_phi):
#   """
#   The structure function increment ratio β_D = \Delta D_{\phi} / D_{\phi}^{theory}(size / 2)
#   DOI: 10.1080/09500340.2019.1596323
#   """
#   w = 0.9955
#   wvl = simulation.source.wvl
#   size = simulation.grid.size
#   phase_screen = simulation.atmosphere.phase_screens[-1]
#   fl = 1 / (3 ** phase_screen["subharmonics"] * size)
#   r0 = Cn2_to_r0(phase_screen["Cn2"], phase_screen["length"], wvl)
#   psd_n = simulation.atmosphere.model
#   psd_phi_f = lambda f: psd_k_to_f(psd_n_to_phi(psd_n, phase_screen["length"], wvl))(f, phase_screen["Cn2"], phase_screen["l0"], phase_screen["L0"])
#   return (w * fl**2 * (8 * psd_phi_f(cp.array([np.sqrt(2) * fl], dtype=np.float32)) + 4 * psd_phi_f(cp.array([fl], dtype=np.float32))) * (1 - np.cos(np.pi * fl * size)) / structure_function_phi(cp.array([size / 2], dtype=np.float32), r0, phase_screen["l0"], phase_screen["L0"])).get()[0]

# def appropriate_subharmonics(simulation, alpha_Phi):
#   "DOI: 10.1080/09500340.2019.1596323"
#   L0 = simulation.atmosphere.phase_screens[-1]["L0"]
#   size = simulation.grid.size
#   return 1 / np.log(3) * np.log(L0 / 2 / size * (alpha_Phi**(-5/6) - 1)**(-1/2))
