import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from sunpy.map import Map
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from matplotlib.colors import LogNorm, ListedColormap

print("Starting Forward Model of Solar Coronal Thomson Scattering")

# Load custom colormap
def load_scientific_colormap(filepath):
    data = np.loadtxt(filepath)
    return ListedColormap(data, name='scientific_colormap', N=data.shape[0])

colormap_path = "/Users/cgilbert/vscode/fluxons/ScientificColourMaps8/romaO/romaO.txt"
scientific_cmap = load_scientific_colormap(colormap_path)
print("Custom colormap loaded.")

# Constants
R_sun = 6.957e10
AU = 1.496e13
sigma_T = 6.652e-25
I_solar_surface = 1.0
print("Constants initialized.")

# Simulation parameters
observer_distance = AU
fov = 32.
viggy = 5.75
npix = 4000
nz = 500
z_max = 10 * R_sun
lower_bound = 5.75
upper_bound = 32.0
print("Simulation parameters set.")

# Output flags
CALC_TOTAL_INTENSITY = True
CALC_POLARIZED_BRIGHTNESS = True
CALC_STOKES_PARAMETERS = True

# Coordinate grid
theta = np.linspace(-fov, fov, npix) * R_sun
X, Y = np.meshgrid(theta, theta)
impact_parameter = np.sqrt(X**2 + Y**2)
position_angle = np.arctan2(Y, X)
z = np.linspace(-z_max, z_max, nz)
print("Coordinate grid created.")

# Electron density model
def electron_density(r):
    r_sr = r / R_sun
    n0 = 4.2e8
    return n0 * 10**(4.32 / r_sr)

# Thomson geometry factors
def thomson_geometry(r, z):
    cos_chi = z / r
    sin2_chi = 1 - cos_chi**2
    return 1 + cos_chi**2, sin2_chi

# Solar intensity drop-off
def incident_solar_intensity(r):
    omega_sun = np.pi * (R_sun / r)**2
    return I_solar_surface * omega_sun / np.pi

# Initialize outputs
B_total = np.zeros((npix, npix))
B_polarized = np.zeros((npix, npix))
Polarization_angle = np.zeros((npix, npix))
print("Output arrays initialized.")

# Simulation loop
print("Starting simulation loop...")
from tqdm import tqdm
for ix in tqdm(range(npix)):
    # if ix % 50 == 0:
        # print(f"Processing row {ix+1}/{npix}")

    rho_row = impact_parameter[ix, :][:, np.newaxis]
    r_LOS = np.sqrt(rho_row**2 + z[np.newaxis, :]**2)
    ne = electron_density(r_LOS)

    G_tot, G_pol = thomson_geometry(r_LOS, z[np.newaxis, :])
    solar_intensity = incident_solar_intensity(r_LOS)
    factor = ne * sigma_T / (4 * np.pi) * solar_intensity / r_LOS**2

    B_total[ix, :] = simpson(factor * G_tot, z, axis=1)
    B_polarized[ix, :] = simpson(factor * G_pol, z, axis=1)
    Polarization_angle[ix, :] = 0.5 * np.arctan2(
        simpson(factor * G_pol * np.sin(2 * position_angle[ix, :][:, None]), z, axis=1),
        simpson(factor * G_pol * np.cos(2 * position_angle[ix, :][:, None]), z, axis=1)
    )
print("Simulation loop completed.")

# Masks
inner_mask = impact_parameter <= R_sun * lower_bound
outer_mask = impact_parameter >= R_sun * upper_bound
combined_mask = inner_mask | outer_mask

B_total[combined_mask] = np.nan
B_polarized[combined_mask] = np.nan
Polarization_angle[combined_mask] = np.nan
print("Masks applied to data arrays.")

Polarization_fraction = B_polarized / B_total
Polarization_fraction[combined_mask] = np.nan

# Plot extent in solar radii
extent = [-fov, fov, -fov, fov]

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

# Shared color normalization for brightness plots
brightness_norm = LogNorm(vmin=np.nanmin([B_total, B_polarized]), vmax=np.nanmax([B_total, B_polarized]))

# Total Brightness
im1 = axs[0, 0].imshow(B_total, origin='lower', extent=extent, cmap='plasma', norm=brightness_norm)
axs[0, 0].set_title('Log Total Brightness')

# Polarized Brightness
im2 = axs[0, 1].imshow(B_polarized, origin='lower', extent=extent, cmap='plasma', norm=brightness_norm)
axs[0, 1].set_title('Log Polarized Brightness (pB)')

# Add shared colorbar for brightness plots
fig.colorbar(im1, ax=axs[0, 0], label='Intensity')
fig.colorbar(im2, ax=axs[0, 1], label='Intensity')

# Polarization Angle
im3 = axs[1, 0].imshow(Polarization_angle, origin='lower', extent=extent, cmap=scientific_cmap)
axs[1, 0].set_title('Linear Polarization Angle')
fig.colorbar(im3, ax=axs[1, 0], label='Radians')

# Polarization Fraction
im4 = axs[1, 1].imshow(Polarization_fraction, origin='lower', extent=extent, cmap='viridis')
axs[1, 1].set_title('Polarization Fraction (pB/tB)')
fig.colorbar(im4, ax=axs[1, 1])

# Draw Sun circle
for ax in axs.flat:
    ax.set_facecolor('black')
    sun_circle = plt.Circle((0, 0), 1, color='yellow', fill=True, alpha=0.8)
    ax.add_artist(sun_circle)

# Axis labels
for ax in axs[1, :]:
    ax.set_xlabel('Solar Radii')
for ax in axs[:, 0]:
    ax.set_ylabel('Solar Radii')

plt.tight_layout()
plt.show()
print("Plots displayed successfully.")