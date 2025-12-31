import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import numpy as np
from astropy import units as u

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
method = "fluxel" # "voronoi" or "fluxel" or "vertex"


def do_fluxlight(flux_world_file, save=True):
    flux_world = read_flux_world(flux_world_file)
    # flux_world.plot_all()

    world_path = flux_world.out_dir
    light_path = world_path.replace("world","light")
    import os
    if not os.path.exists(light_path):
        os.makedirs(light_path)

    print(world_path)
    print(light_path)
    # flux_world.plot_fluxon_id()
    nz = 400
    # Run the Thomson scattering simulation, informing the model with the flux world
    print(f"Starting Fluxlight {method = }...")
    results = simulate_thomson_scattering(
        npix=200,
        nz=nz,
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
    axs[0, 0].set_title("Log Total Brightness (tB)")
    fig.colorbar(im1, ax=axs[0, 0], label="Intensity")

    im2 = axs[0, 1].imshow(B_polarized.to_value(), origin="lower", extent=extent_val, cmap="plasma", norm=brightness_norm)
    axs[0, 1].set_title("Log Polarized Brightness (pB)")
    fig.colorbar(im2, ax=axs[0, 1], label="Intensity")

    # Build a SunPy meta dictionary and apply RHEF to the column density map:
    cd = Column_Density.to_value()
    nx, ny = cd.shape
    fov_rsun = fov.to_value()
    pixscale_rsun = (2 * fov_rsun) / nx

    from sunpy.coordinates import sun
    from astropy import constants as const
    from astropy import units as u
    angular_radius = sun.angular_radius().to(u.arcsec)
    rsun_arcsec = angular_radius.value
    pixscale_arcsec = pixscale_rsun * rsun_arcsec

    meta = {
        "cdelt1": pixscale_arcsec,
        "cdelt2": pixscale_arcsec,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "crpix1": nx / 2,
        "crpix2": ny / 2,
        "crval1": 0.0,
        "crval2": 0.0,
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "rsun_ref": rsun_arcsec,
        "dsun_obs": const.au.to_value(u.m),
        "hgln_obs": 0.0,
        "hglt_obs": 0.0,
        "obsrvtry": "Earth",
        "naxis1": nx,
        "naxis2": ny,
    }

    density_map = Map(cd, meta)
    filtered_density = rhef(density_map, application_radius=1.05*u.R_sun, upsilon="none")
    im3 = axs[1, 0].imshow(filtered_density.data, origin="lower", extent=extent_val, cmap="magma")
    axs[1, 0].set_title("RHEF-filtered Column Density")
    fig.colorbar(im3, ax=axs[1, 0], label="log(cm⁻²)")

    im4 = axs[1, 1].imshow(Polarization_fraction, origin="lower", extent=extent_val, cmap="viridis")
    axs[1, 1].set_title("Polarization Fraction (pB/tB)")
    fig.colorbar(im4, ax=axs[1, 1])

    for ax in axs.flat:
        sun_circle = patches.Circle((0, 0), 1, color="yellow", alpha=1.0, zorder=10)
        ax.add_patch(sun_circle)
        ax.set_facecolor("black")

        ax.set_xlabel("Solar Radii")
        ax.set_ylabel("Solar Radii")

    plt.suptitle(f"Forward Modeling of FLUX World CR {flux_world.cr}")
    plt.tight_layout()
    if save:
        lightpath = f"{light_path}/cr{flux_world.cr}_fluxlight_light_{method}_{nz}.png"
        print(lightpath)
        plt.savefig(lightpath)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    print("\n\n\nLET THERE BE LIGHT\n\n", flush=True)
    import sys

    if len(sys.argv) > 1:
        the_world_file = sys.argv[1]
    else:
        the_world_file = def_world_file

    do_fluxlight(the_world_file)

    # f_world = read_flux_world(the_world_file)
    # f_world.plot_all()
