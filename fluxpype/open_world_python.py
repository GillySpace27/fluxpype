from fluxpype.pipe_helper import read_flux_world

f_world = read_flux_world("/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Newtest/data/cr2225/world/cr2225_f2000_hmi_relaxed_s800.flux")

print(f_world, "\n")
print(f_world.fluxons, "\n")
print(f_world.concentrations, "\n")
