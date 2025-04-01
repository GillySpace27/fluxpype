import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from matplotlib.colors import LogNorm, ListedColormap
from tqdm import tqdm
from joblib import Parallel, delayed
from astropy import units as u
from astropy.constants import L_sun, R_sun


def ensure_rsun_quantity(val):
    return val * u.R_sun if not isinstance(val, u.Quantity) else val.to(u.R_sun)


# Thomson cross-section in cm^2
sigma_T = 6.652e-25 * u.cm**2

# Specific intensity of the sun center
I_sp_solar_surface = (L_sun / (4 * np.pi**2 * R_sun**2)).to(u.erg / u.cm**2 / u.s / u.sr, equivalencies=u.dimensionless_angles())
u.I_sun = u.def_unit("I_sun", I_sp_solar_surface, format={"latex": r"I_{\odot}", "unicode": "I☉"})
I_sp_solar_surface = 1.0 * u.I_sun

from astropy import units as u


@u.quantity_input(r=u.R_sun)
def default_electron_density(r):
    """
    Return coronal electron density in cm^-3, assuming r is in solar radii.
    """
    n0 = 4.2e8 * u.cm**-3
    return (n0 * 10 ** (4.32 / r.to_value(u.R_sun))).to(u.cm**-3)


@u.quantity_input(r=u.R_sun, z=u.R_sun)
def thomson_geometry(r, z):
    """
    r, z are in solar radii. z is along line-of-sight, r_LOS = sqrt(rho^2 + z^2).
    """
    cos_chi = z / r
    # 1 + cos^2(chi), sin^2(chi) = 1 - cos^2(chi)
    return (1 + cos_chi**2), (1 - cos_chi**2)


@u.quantity_input(r=u.R_sun)
def incident_solar_intensity(r):
    """
    Dimensionless solar intensity ~ 1 / r^2, ignoring physical constants.
    """
    return I_sp_solar_surface / (r**2)


def simulate_thomson_scattering(
    npix=500, nz=500, fov=3.0, lower_bound=1.01, upper_bound=3.0,
    flux_world=None, influence_length=1.0, z_max = 10.0, parallel=False
):
    """
    Simulate Thomson scattering brightness and polarization in the solar corona.

    Parameters
    ----------
    npix : int
        Number of pixels along one dimension of the image grid.
    nz : int
        Number of integration steps along the line of sight.
    z_max : float or Quantity
        Maximum distance along the line of sight (LOS) in units of solar radii.
    fov : float or Quantity
        Field of view from the center to edge, interpreted in units of solar radii.
    lower_bound : float or Quantity
        Inner masking radius; emission below this radius will be excluded.
    upper_bound : float or Quantity
        Outer masking radius; emission beyond this radius will be excluded.
    flux_world : FluxWorld, optional
        If provided, uses the fluxon-based density model.
    influence_length : float or Quantity
        Length scale of influence of a fluxon's density enhancement.

    Returns
    -------
    dict
        Dictionary containing 2D arrays of:
        - "B_total": Total brightness.
        - "B_polarized": Polarized brightness.
        - "Polarization_angle": Angle of polarization.
        - "Polarization_fraction": Degree of polarization.
        - "X", "Y": Coordinate grids in solar radii.
        - "impact_parameter", "position_angle": Image plane parameters.
        - "fov": Field of view in solar radii.
    """
    # Convert parameters to quantities if they are not already
    fov = ensure_rsun_quantity(fov)
    z_max = ensure_rsun_quantity(z_max)
    lower_bound = ensure_rsun_quantity(lower_bound)
    upper_bound = ensure_rsun_quantity(upper_bound)
    influence_length = ensure_rsun_quantity(influence_length)

    # Create a coordinate grid in solar radii
    side = np.linspace(-fov, fov, npix)
    X, Y = np.meshgrid(side, side)
    impact_parameter = np.sqrt(X**2 + Y**2)
    position_angle = np.arctan2(Y, X)

    # Range for line-of-sight integration (in solar radii)
    z_array = np.linspace(-z_max.to_value(u.R_sun), z_max.to_value(u.R_sun), nz) * u.R_sun

    # Initialize outputs
    B_total = np.zeros((npix, npix)) * (u.erg / u.cm**2 / u.s / u.sr)
    B_polarized = np.zeros((npix, npix)) * (u.erg / u.cm**2 / u.s / u.sr)
    Polarization_angle = np.zeros((npix, npix)) #* u.rad

    def process_row(ix):
        # Impact param row, pos angle row
        rho_row = impact_parameter[ix, :]
        pos_ang_row = position_angle[ix, :]

        # r_LOS: sqrt(rho^2 + z^2) for each pixel's line of sight
        r_LOS = np.sqrt(rho_row[:, None]**2 + z_array[None, :]**2)

        # Electron density along LOS
        if flux_world is not None:
            # Build the 3D coords in solar radii
            X_row = X[ix, :].reshape(-1, 1)
            Y_row = Y[ix, :].reshape(-1, 1)
            Z_grid = z_array[None, :]
            coords = np.stack([
                np.repeat(X_row, nz, axis=1),
                np.repeat(Y_row, nz, axis=1),
                np.repeat(Z_grid, rho_row.shape[0], axis=0),
            ], axis=-1)  # (npix, nz, 3)
            coords_flat = coords.reshape(-1, 3)
            ne_flat = flux_world.compute_electron_density(
                coords_flat, influence_length=influence_length
            )
            ne = ne_flat.reshape(rho_row.shape[0], nz)  # * u.cm**-3
        else:
            ne = default_electron_density(r_LOS)

        # Thomson geometry
        G_tot, G_pol = thomson_geometry(r_LOS, z_array[None, :])
        solar_int = incident_solar_intensity(r_LOS)
        factor = ne * sigma_T / (4.0 * np.pi) * solar_int / (r_LOS**2)

        # Integrate along the LOS
        intensity_unit = (u.erg / u.cm**2 / u.s / u.sr)
        B_total_row = simpson((factor * G_tot).value, z_array.value, axis=1) * intensity_unit
        B_polarized_row = simpson((factor * G_pol).value, z_array.value, axis=1) * intensity_unit
        num = simpson((factor * G_pol * np.sin(2 * pos_ang_row[:, None])).value, z_array.value, axis=1)
        den = simpson((factor * G_pol * np.cos(2 * pos_ang_row[:, None])).value, z_array.value, axis=1)
        Pol_ang_row = 0.5 * np.arctan2(num, den) * u.radian
        # dev = Pol_ang_row - pos_ang_row
        # dev = dev.to_value()
        # dev = np.angle(np.exp(1j * (Pol_ang_row - pos_ang_row).to_value()))
        dev = np.angle(np.exp(1j * (Pol_ang_row).to_value()))
        dev = Pol_ang_row.to_value()
        # dev[dev > 3.1] = dev[dev>3.1] - np.pi
        # dev[dev < -3.1] = dev[dev<-3.1] + np.pi
        return B_total_row, B_polarized_row, dev

    if parallel:
        results_list = Parallel(n_jobs=-1)(
            delayed(process_row)(ix) for ix in tqdm(range(npix), desc="Simulating Thomson (Solar-Radii)")
        )
    else:
        results_list = []
        for ix in tqdm(range(npix), desc="Simulating Thomson (Solar-Radii, Serial)"):
            results_list.append(process_row(ix))

    for ix, (B_tot_row, B_pol_row, Pol_ang_row) in enumerate(results_list):
        B_total[ix, :] = B_tot_row
        B_polarized[ix, :] = B_pol_row
        Polarization_angle[ix, :] = Pol_ang_row

    # Mask out areas inside lower_bound or outside upper_bound
    inner_mask = impact_parameter <= lower_bound
    outer_mask = impact_parameter >= upper_bound
    combined_mask = inner_mask | outer_mask
    B_total[combined_mask] = np.nan
    B_polarized[combined_mask] = np.nan
    Polarization_angle[combined_mask] = np.nan

    Polarization_fraction = B_polarized / B_total
    Polarization_fraction[combined_mask] = np.nan

    return {
        "B_total": B_total,
        "B_polarized": B_polarized,
        "Polarization_angle": Polarization_angle,
        "Polarization_fraction": Polarization_fraction,
        "X": X,
        "Y": Y,
        "impact_parameter": impact_parameter,
        "position_angle": position_angle,
        "fov": fov,
    }


def load_scientific_colormap(filepath):
    data = np.loadtxt(filepath)
    return ListedColormap(data, name="scientific_colormap", N=data.shape[0])


if __name__ == "__main__":


    import sys

    if len(sys.argv) > 1:
        the_world_file = sys.argv[1]
    else:
        the_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/fluxlight/data/cr2230/world/cr2230_f1000_hmi_relaxed_s800.flux"

    from fluxpype.science.open_world_python import read_flux_world
    f_world = read_flux_world(the_world_file)
    f_world.plot_all(save=True)








    # res = simulate_thomson_scattering(parallel=True)











    # print("Result shape:", res["B_total"].shape)
    # plt.imshow(
    #     res["B_total"].value,
    #     origin="lower",
    #     cmap="plasma",
    #     norm=LogNorm(vmin=np.nanmin(res["B_total"].value), vmax=np.nanmax(res["B_total"].value)),
    # )
    # plt.gca().set_facecolor("black")
    # plt.title("B_total in from Thomson Scattering")
    # plt.colorbar(label=res["B_total"].unit)

    # if filename is not None:
    #     from os.path import basename
    #     import re

    #     self.name = basename(filename)
    #     self.out_dir = os.path.join(filename.split("data/cr")[0], "imgs", "world")
    #     match = re.search(r"cr(\d{4})", filename)
    #     if match:
    #         self.cr = match.group(1)
    #     match_f = re.search(r"_f(\d+)_", filename)
    #     if match_f:
    #         self.nflx = match_f.group(1)
