import os
import pickle
import matplotlib
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


def keynote_figs():
    matplotlib.rcParams["lines.color"] = "white"
    matplotlib.rcParams["patch.edgecolor"] = "white"
    matplotlib.rcParams["text.color"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "black"
    matplotlib.rcParams["axes.edgecolor"] = "white"
    matplotlib.rcParams["axes.labelcolor"] = "white"
    matplotlib.rcParams["xtick.color"] = "white"
    matplotlib.rcParams["ytick.color"] = "white"
    matplotlib.rcParams["grid.color"] = "white"
    matplotlib.rcParams["figure.facecolor"] = "black"
    matplotlib.rcParams["figure.edgecolor"] = "black"
    matplotlib.rcParams["savefig.facecolor"] = "black"
    matplotlib.rcParams["savefig.edgecolor"] = "black"
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["lines.linewidth"] = 1.5
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["text.usetex"] = False


def light_figs():
    matplotlib.rcParams["lines.color"] = "black"
    matplotlib.rcParams["patch.edgecolor"] = "black"
    matplotlib.rcParams["text.color"] = "black"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["axes.edgecolor"] = "black"
    matplotlib.rcParams["axes.labelcolor"] = "black"
    matplotlib.rcParams["xtick.color"] = "black"
    matplotlib.rcParams["ytick.color"] = "black"
    matplotlib.rcParams["grid.color"] = "black"
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["figure.edgecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.edgecolor"] = "white"
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["lines.linewidth"] = 1.5
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["text.usetex"] = False


keynote_figs()
# light_figs()


# Load the flux world data
# flux_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Force_test/data/cr2265/world/cr2265_f1002_hmi_relaxed_s400.flux"
# flux_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Force_test/data/cr2268/world/cr2268_f500_hmi_relaxed_s4000.flux"
# flux_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Relaxation_Troubleshooting/data/cr2229/world/cr2229_f20_hmi_relaxed_s2000.flux"
# def_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/Relaxation_Troubleshooting/data/cr2229/world/cr2229_f400_hmi.flux"
def_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/fluxlight/data/cr2150/world/cr2150_f1000_hmi_relaxed_s300.flux"
def_method = "fluxel"  # "voronoi" or "soft_voronoi" or "fluxel" or "vertex"
def_influence_length = 1.0
def_nz = 301
def_nxy = 301
def_fov = 215.0

from fluxpype.pipe_helper import configurations

configs = configurations()

method = configs.get("light_method", def_method)
influence_length = configs.get("influence_length", def_influence_length)
nz = configs.get("nz", def_nz)
nxy = configs.get("nxy", configs.get("npix", def_nxy))
fov = configs.get("fov", def_fov)

def do_fluxlight(flux_world_file, save=True, force=False):
    results_file = flux_world_file.replace(".flux", ".light")

    if os.path.exists(results_file) and not force:
        print(f"Loading cached Fluxlight results from {results_file}")
        with open(results_file, "rb") as f:
            results = pickle.load(f)
        return results
    flux_world = read_flux_world(flux_world_file)
    # flux_world.plot_all()

    world_path = flux_world.out_dir
    light_path = world_path.replace("world","light")
    if not os.path.exists(light_path):
        os.makedirs(light_path)

    print(world_path)
    print(light_path)
    # flux_world.plot_fluxon_id()

    # Run the Thomson scattering simulation, informing the model with the flux world
    print(f"Starting Fluxlight {method = }...")
    results = simulate_thomson_scattering(
        npix=nxy,
        nz=nz,
        fov=fov,
        flux_world=flux_world,
        lower_bound=1.05,
        upper_bound=100.0,
        parallel = True,
        influence_length=influence_length,
        scale=20,
        method=method,
        # voronoi_blur=0.04,
        # voronoi_samples=9
    )
    results["nz"] = nz
    results["nxy"] = nxy
    results["fov"] = fov
    results["world"] = flux_world
    results["light_path"] = light_path
    print("Fluxlight Complete.")
    # Cache results to disk
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved Fluxlight results to {results_file}")
    return results

def densplot(results, ax):
    Column_Density = results["Column_Density"]

    # Build a SunPy meta dictionary and apply RHEF to the column density map:
    cd = Column_Density.to_value()
    fov = results["fov"]

    # Set the plot extent (in solar radii)
    fov_val = fov.to_value()
    extent_val = [-fov_val, fov_val, -fov_val, fov_val]

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
    filtered_density = rhef(density_map, application_radius=1.05 * u.R_sun, upsilon="none")
    # filtered_density = density_map
    im3 = ax.imshow(filtered_density.data, origin="lower", extent=extent_val, cmap="magma")
    ax.set_title("RHEF-filtered Column Density")
    fig = ax.figure
    fig.colorbar(im3, ax=ax, label="log(cm⁻²)")


def quadplot(results, save=True):
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

    densplot(results, axs[1,0])

    im4 = axs[1, 1].imshow(Polarization_fraction, origin="lower", extent=extent_val, cmap="viridis")
    axs[1, 1].set_title("Polarization Fraction (pB/tB)")
    fig.colorbar(im4, ax=axs[1, 1])

    for ax in axs.flat:
        sun_circle = patches.Circle((0, 0), 1, color="yellow", alpha=1.0, zorder=10)
        ax.add_patch(sun_circle)
        ax.set_facecolor("black")

        ax.set_xlabel("Solar Radii")
        ax.set_ylabel("Solar Radii")

    plt.suptitle(f"Forward Modeling of FLUX World CR {results["world"].cr}")
    plt.tight_layout()
    if save:
        lightpath = f"{results["light_path"]}/cr{results["world"].cr}_fluxlight_light_{method}_{results["nz"]}.png"
        print(lightpath)
        plt.savefig(lightpath)
        plt.close(fig)
    else:
        plt.show()


def triplot(results, fov_crop=None, save=True):
    # Retrieve simulation outputs and grid information
    B_total = results["B_total"]
    B_polarized = results["B_polarized"]
    Polarization_angle = results["Polarization_angle"]
    Column_Density = results["Column_Density"]
    Polarization_fraction = results["Polarization_fraction"]
    X = results["X"]
    Y = results["Y"]

    # if fov is None:
    fov = results["fov"]

    # Set the uniform colormap for the polarization angle
    colormap_path = "/Users/cgilbert/vscode/fluxons/ScientificColourMaps8/romaO/romaO.txt"
    scientific_cmap = load_scientific_colormap(colormap_path)
    brightness_norm = LogNorm(vmin=np.nanmin([B_total, B_polarized]), vmax=np.nanmax([B_total, B_polarized]))

    # Set the plot extent (in solar radii)
    try:
        fov_val = fov.to_value()
    except AttributeError:
        fov_val = fov

    extent_val = [-fov_val, fov_val, -fov_val, fov_val]

    # Create a 1x3 grid of plots to visualize the results
    fig, axs = plt.subplots(1,3, figsize=(14, 4.5), sharex=True, sharey=True)
    axes = axs.flatten()

    im1 = axes[0].imshow(B_total.to_value(), origin="lower", extent=extent_val, cmap="plasma", norm=brightness_norm)
    axes[0].set_title("Total Brightness (tB)")
    fig.colorbar(im1, ax=axes[0], label="Intensity")

    im2 = axes[1].imshow(B_polarized.to_value(), origin="lower", extent=extent_val, cmap="plasma", norm=brightness_norm)
    axes[1].set_title("Polarized Brightness (pB)")
    fig.colorbar(im2, ax=axes[1], label="Intensity")

    # im3 = densplot(results, axs[1,0])

    im4 = axes[2].imshow(Polarization_fraction, origin="lower", extent=extent_val, cmap="viridis")
    axes[2].set_title("Polarization Fraction (pB/tB)")
    fig.colorbar(im4, ax=axes[2])

    for ax in axes:
        sun_circle = patches.Circle((0, 0), 1, color="yellow", alpha=1.0, zorder=10)
        ax.add_patch(sun_circle)
        ax.set_facecolor("black")

        ax.set_xlabel("Solar Radii")
        ax.set_ylabel("Solar Radii")
        ax.set_xlim([-fov_crop, fov_crop])
        ax.set_ylim([-fov_crop, fov_crop])

    plt.suptitle(f"Forward Modeling of FLUX CR {results["world"].cr}, R={results["influence_length"]}")
    plt.tight_layout()
    if save:
        lightpath = f"{results["light_path"]}/cr{results["world"].cr}_fluxlight_light_{method}_{results["nz"]}_{fov_crop}_{results["influence_length"].to_value()}.png"
        print(lightpath)
        plt.savefig(lightpath, dpi=250)
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

    force = "--force" in sys.argv or True
    results = do_fluxlight(the_world_file, force=force)


    triplot(results, 100) # * u.R_sun)
    triplot(results, 50 ) #* u.R_sun)
    triplot(results, 25 ) #* u.R_sun)
    triplot(results, 10 ) #* u.R_sun)

    # f_world = read_flux_world(the_world_file)
    # f_world.plot_all()
