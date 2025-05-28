import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import numpy as np

# Import the modules (adjust the import paths as needed)
from fluxpype.science.open_world_python import read_flux_world
from fluxpype.science.thomson_scattering_flux import simulate_thomson_scattering, load_scientific_colormap
from sunkit_image.radial import rhef
from sunpy.map import Map

# Load the flux world data
# flux_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Force_test/data/cr2265/world/cr2265_f1002_hmi_relaxed_s400.flux"
# flux_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Force_test/data/cr2268/world/cr2268_f500_hmi_relaxed_s4000.flux"
# flux_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Relaxation_Troubleshooting/data/cr2229/world/cr2229_f20_hmi_relaxed_s2000.flux"
# def_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Relaxation_Troubleshooting/data/cr2229/world/cr2229_f400_hmi.flux"
def_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/fluxlight/data/cr2150/world/cr2150_f1000_hmi_relaxed_s300.flux"
method = "vertex"


def do_fluxlight(flux_world_file, save=True):
    flux_world = read_flux_world(flux_world_file)
    # flux_world.plot_all()

    world_path = flux_world.out_dir
    light_path = world_path.replace("world","light")
    import os
    if not os.path.exists(light_path):
        os.makedirs(light_path)

    # print(flux_world)
    # flux_world.plot_fluxon_id()

    # Run the Thomson scattering simulation, informing the model with the flux world
    print("Starting Fluxlight...")
    results = simulate_thomson_scattering(
        npix=250,
        nz=250,
        fov=3.0,
        flux_world=flux_world,
        lower_bound=1.05,
        upper_bound=3.0,
        parallel = True,
        influence_length=0.05,
        scale=500,
        method=method
    )
    print("Fluxlight Complete.")
    # Retrieve simulation outputs and grid information
    B_total = results["B_total"]
    B_polarized = results["B_polarized"]
    Polarization_angle = results["Polarization_angle"]
    Column_Density = results["Column_Density"]
    Polarization_fraction = results["Polarization_fraction"]
    X = results["X"]
    Y = results["Y"]
    fov = results["fov"]

    # Set the uniform colormap for the polarization angle
    colormap_path = "/Users/cgilbert/vscode/fluxons/ScientificColourMaps8/romaO/romaO.txt"
    scientific_cmap = load_scientific_colormap(colormap_path)
    brightness_norm = LogNorm(vmin=np.nanmin([B_total, B_polarized]), vmax=np.nanmax([B_total, B_polarized]))

    # Set the plot extent (in solar radii)
    fov_val = fov.to_value()
    extent_val = [-fov_val, fov_val, -fov_val, fov_val]

    # Create a 2x2 grid of plots to visualize the results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    im1 = axs[0, 0].imshow(B_total.to_value(), origin="lower", extent=extent_val, cmap="plasma", norm=brightness_norm)
    axs[0, 0].set_title("Log Total Brightness")
    fig.colorbar(im1, ax=axs[0, 0], label="Intensity")

    im2 = axs[0, 1].imshow(B_polarized.to_value(), origin="lower", extent=extent_val, cmap="plasma", norm=brightness_norm)
    axs[0, 1].set_title("Log Polarized Brightness (pB)")
    fig.colorbar(im2, ax=axs[0, 1], label="Intensity")

    # Placeholder for plane-of-sky density: show redundant brightness
    cd = Column_Density.to_value()
    im3 = axs[1, 0].imshow(rhef(Map(cd)), origin="lower", extent=extent_val, cmap="plasma")
    axs[1, 0].set_title("Redundant Total Brightness")
    fig.colorbar(im3, ax=axs[1, 0], label="Intensity")

    im4 = axs[1, 1].imshow(Polarization_fraction, origin="lower", extent=extent_val, cmap="viridis")
    axs[1, 1].set_title("Polarization Fraction (pB/tB)")
    fig.colorbar(im4, ax=axs[1, 1])

    for ax in axs.flat:
        sun_circle = patches.Circle((0, 0), 1, color="yellow", alpha=0.8, zorder=10)
        ax.add_patch(sun_circle)
        ax.set_facecolor("black")

        ax.set_xlabel("Solar Radii")
        ax.set_ylabel("Solar Radii")

    plt.suptitle("Forward Modeling of FLUX World")
    plt.tight_layout()
    if save:
        plt.savefig(f"{light_path}/cr{flux_world.cr}_fluxlight_light_{method}.png")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    print("\n\n\nLET THERE BE LIGHT\n\n")
    import sys

    if len(sys.argv) > 1:
        the_world_file = sys.argv[1]
    else:
        the_world_file = def_world_file

    do_fluxlight(the_world_file)

    # f_world = read_flux_world(the_world_file)
    # f_world.plot_all()
