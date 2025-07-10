import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import astropy.units as u

"""
This module processes FLUX world output data to create visualizations of flux concentrations and fluxons in 3D.
It defines the Fluxon class for individual fluxon properties and the FluxWorld class to encapsulate
flux concentrations, fluxons, and associated visualization and computation methods (electron density,
cross-sectional areas, etc.).

Dependencies:
    - numpy, matplotlib, scipy, astropy, and mpl_toolkits

Usage:
    The module can be executed as a script to load a FLUX world file and generate visualizations.
"""


class Fluxon:
    """
    Represents an individual fluxon with its endpoints, flux value, spatial coordinates,
    computed cross-sectional areas, and radial distances. Also determines whether the fluxon
    is open, closed, or of another type based on its endpoints.
    """
    def __init__(self, fid, start_fc, end_fc, flux, x_coords, y_coords, z_coords):
        """
        Initialize a Fluxon object.

        Parameters:
            fid (int): Unique fluxon identifier.
            start_fc (int): Starting flux concentration indicator (-1 indicates open).
            end_fc (int): Ending flux concentration indicator (-2 indicates open).
            flux (float): Flux value associated with the fluxon.
            x_coords, y_coords, z_coords (list or array): Lists of coordinates for the fluxon's vertices.
        """
        import numpy as np

        self.id = fid
        self.start_fc = start_fc
        self.end_fc = end_fc
        self.flux = flux
        self.x_coords = np.array(x_coords)
        self.y_coords = np.array(y_coords)
        self.z_coords = np.array(z_coords)
        self.areas = None  # To be filled with computed cross-sectional areas
        self.radius = np.sqrt(self.x_coords**2 + self.y_coords**2 + self.z_coords**2)
        # Flip the arrays if the fluxon is reversed so that the smallest radius is first
        if self.radius[0] > self.radius[-1]:
            self.x_coords = self.x_coords[::-1]
            self.y_coords = self.y_coords[::-1]
            self.z_coords = self.z_coords[::-1]
            self.radius = self.radius[::-1]
        self.kind = self.determine_kind()
        self.disabled = False

    def determine_kind(self):
        """
        Determine the type of the fluxon based on its endpoints.

        Returns:
            str: 'open' if exactly one endpoint indicates open, 'closed' if both are not open,
                 or 'other' for any other combination.
        """
        st_open = self.start_fc == -1
        en_open = self.end_fc == -2
        if st_open + en_open == 1:
            self.kind = "open"
        elif st_open + en_open == 0:
            self.kind = "closed"
        else:
            self.kind = "other"
        return self.kind

    def get_vertices(self):
        """
        Get the spatial vertices of the fluxon.

        Returns:
            np.ndarray: An (N x 3) array where each row is the (x, y, z) coordinate of a vertex.
        """
        return np.column_stack((self.x_coords, self.y_coords, self.z_coords))

    def get_radius(self):
        """
        Get the computed radial distances for the fluxon vertices.

        Returns:
            np.ndarray: The radial distances computed from the fluxon's coordinates.
        """
        return self.radius

    def set_areas(self, areas, raw_radius=None):
        """
        Set the cross-sectional areas for the fluxon, optionally interpolating them onto the fluxon's internal radius grid.

        Parameters:
            areas (array-like): Computed areas corresponding to each vertex.
            raw_radius (array-like, optional): Raw radius values corresponding to the provided areas. If provided, the areas
                                               will be interpolated onto the fluxon's internal radius grid (self.radius).
        """
        import numpy as np
        if raw_radius is not None:
            # Interpolate the provided area data onto the fluxon's internal radius grid.
            areas = np.interp(self.radius, raw_radius, areas, left=areas[0], right=areas[-1])

        # Ensure the fluxon is ordered from the photosphere outward by checking the radius array.
        if self.radius[0] > self.radius[-1]:
            self.x_coords = self.x_coords[::-1]
            self.y_coords = self.y_coords[::-1]
            self.z_coords = self.z_coords[::-1]
            self.radius = self.radius[::-1]
            areas = np.array(areas)[::-1]

        self.areas = areas
        self.fr = self.compute_expansion_factor(self.areas, self.radius)

        # Check for suspiciously constant area with height.
        const_threshold = 0.01  # relative variation threshold
        mean_area = np.mean(self.areas)
        if mean_area != 0 and (np.ptp(self.areas) / mean_area) < const_threshold:
            self.disabled = True
            # print(f"Fluxon {self.id} rejected: constant area with height.")
        else:
            self.disabled = False

    def __str__(self):
        """
        Return a string representation of the fluxon.

        Returns:
            str: A summary string of the fluxon including id, flux, and number of vertices.
        """
        return f"Fluxon(id={self.id}, flux={self.flux}, n_vertices={len(self.x_coords)})"

    def compute_expansion_factor(self, A, r):
        """
        Compute the expansion factor from an array of areas A and radii r.

        The expansion factor is defined as:
            fr = (A / A[1]) * (r[1]**2 / r**2)

        Parameters:
        A : array-like
            Array of areas.
        r : array-like
            Array of radii.

        Returns:
        numpy.ndarray
            The computed expansion factor.

        Raises:
        ValueError: If A or r have fewer than 2 elements.
        """
        A = np.asarray(A)
        r = np.asarray(r)

        if A.size < 2 or r.size < 2:
            raise ValueError("Input arrays must have at least 2 elements.")

        fr = (A / A[1]) * ((r[1] ** 2) / (r**2))
        return fr


def read_flux_world(filename):
    """
    Reads FLUX world output data from a file and parses it into a structured FluxWorld object.

    Parameters:
        filename (str): Path to the file or directory containing FLUX world data.

    Returns:
        FluxWorld: An object containing parsed flux concentrations and fluxons along with methods
                   for visualization and computation (e.g., electron density, cross-sectional areas).
    """

    print(f"Reading FLUX world file: {filename}")
    class FluxWorld:
        def __init__(self, filename=None):
            self.concentrations = self.FluxConcentrations()
            # Use a dictionary to store individual Fluxon objects (keyed by fluxon id)
            self.fluxons = {}
            self.tree = None
            self.coords = None
            self.densities = None
            self.name = None
            self.cr = "unknown"
            self.nflx = "unknown"
            self.areas_by_fluxon = None
            self.filename = filename or None
            self.flux_area_file = None
            self.segments = None
            if filename is not None:
                from os.path import basename, dirname, join
                from os import listdir
                from pathlib import Path
                import re

                self.name = basename(filename)

                self.out_dir = join(filename.split("data/cr")[0], "imgs", "world")
                self.dat_dir = join(Path(dirname(filename)).parent, "wind")

                match = re.search(r"cr(\d{4})", filename)
                if match:
                    self.cr = match.group(1)
                match_f = re.search(r"_f(\d+)_", filename)
                if match_f:
                    self.nflx = match_f.group(1)

                try:
                    self.flux_area_file = [
                        join(self.dat_dir, pp)
                        for pp in listdir(self.dat_dir)
                        if (pp.endswith("_radial_fr.dat") and self.cr in pp and self.nflx in pp)
                    ][0]
                    pass
                except Exception as e:
                    raise e

        def __str__(self):
            key_color = "\033[94m"  # blue color for keys
            reset_color = "\033[0m"  # reset color
            nm = f" {self.name}" if self.name else ""
            return (
                f"{key_color}FluxWorld{reset_color}{nm}: {len(self.concentrations.ids)} concentrations, "
                f"{len(self.fluxons)} fluxons."
            )

        def plot_all(self, **kwargs):
            print(f"Saving to {self.out_dir}")
            self.plot_all_area_methods(save=True)
            # self.plot_all_fluxon_area_methods(save=True)
            self.plot_fluxon_areas(save=True)
            self.plot_world(save=True)
            self.plot_density(save=True)
            self.plot_fluxon_id(save=True)

        def plot_world(
            self,
            color_by="kind",
            alpha=0.6,
            save=False,
            extent=30.0,
            plot_sphere=True,
            plot_open_fluxons=True,
            plot_closed_fluxons=True,
            plot_concentrations=False,
            plot_vertices=False,
        ):
            """
            Plots flux concentrations and fluxons in 3D using matplotlib.
            """
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            if plot_concentrations:
                cxs = np.array(self.concentrations.x_coords)
                cys = np.array(self.concentrations.y_coords)
                czs = np.array(self.concentrations.z_coords)
                cflux = np.array(self.concentrations.flux_values)
                conc_colors = ["red" if f > 0 else "blue" for f in cflux]
                ax.scatter(cxs, cys, czs, c=conc_colors, s=40, alpha=0.8, marker="o", label="Concentrations")

            all_segments = []
            all_values_for_coloring = []

            for flux in self.fluxons.values():
                # Determine fluxon type based on endpoints
                st_open = flux.start_fc == -1
                en_open = flux.end_fc == -2
                if st_open + en_open == 1:
                    kind = "open"
                    if not plot_open_fluxons:
                        continue
                elif st_open + en_open == 0:
                    kind = "closed"
                    if not plot_closed_fluxons:
                        continue
                else:
                    continue

                x_arr = flux.x_coords
                y_arr = flux.y_coords
                z_arr = flux.z_coords
                flux_value = flux.flux

                num_pts = len(x_arr)
                for j in range(num_pts - 1):
                    p1 = np.array([x_arr[j], y_arr[j], z_arr[j]])
                    p2 = np.array([x_arr[j + 1], y_arr[j + 1], z_arr[j + 1]])
                    all_segments.append([p1, p2])
                    if color_by == "radial_deviation":
                        midpoint = 0.5 * (p1 + p2)
                        radial_vec = midpoint
                        seg_vec = p2 - p1
                        if np.linalg.norm(radial_vec) < 1e-12 or np.linalg.norm(seg_vec) < 1e-12:
                            angle = 0.0
                        else:
                            cos_theta = np.dot(radial_vec, seg_vec) / (
                                np.linalg.norm(radial_vec) * np.linalg.norm(seg_vec)
                            )
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            angle = np.arccos(cos_theta)
                        all_values_for_coloring.append(angle)
                    elif color_by == "flux":
                        all_values_for_coloring.append(flux_value)
                    elif color_by == "kind":
                        all_values_for_coloring.append(kind)
                    else:
                        all_values_for_coloring.append(0.0)

            line_collection = Line3DCollection(all_segments, linewidth=1.5, alpha=alpha)
            if color_by in ["radial_deviation", "flux"]:
                all_values_for_coloring = np.array(all_values_for_coloring)
                line_collection.set_array(all_values_for_coloring)
                line_collection.set_cmap("RdYlBu")
                ax.add_collection3d(line_collection)
                cbar = plt.colorbar(line_collection, ax=ax, pad=0.1, shrink=0.8)
                if color_by == "radial_deviation":
                    cbar.set_label("Angle from radial (radians)")
                elif color_by == "flux":
                    cbar.set_label("Flux value")
            elif color_by == "kind":
                color_mapping = {"open": "green", "closed": "red"}
                segment_colors = [color_mapping.get(k, "black") for k in all_values_for_coloring]
                line_collection.set_color(segment_colors)
                ax.add_collection3d(line_collection)
            else:
                line_collection.set_color("green")
                ax.add_collection3d(line_collection)
            if plot_vertices:
                all_vertex_x = []
                all_vertex_y = []
                all_vertex_z = []
                for flux in self.fluxons.values():
                    if len(flux.x_coords) > 2:
                        all_vertex_x.extend(flux.x_coords[1:-1])
                        all_vertex_y.extend(flux.y_coords[1:-1])
                        all_vertex_z.extend(flux.z_coords[1:-1])
                if all_vertex_x:
                    ax.scatter(all_vertex_x, all_vertex_y, all_vertex_z, c="magenta", s=20, alpha=0.8, label="Vertices")

            u_vals = np.linspace(0, 2 * np.pi, 20)
            v_vals = np.linspace(0, np.pi, 20)
            x_sphere = 1.0 * np.outer(np.cos(u_vals), np.sin(v_vals))
            y_sphere = 1.0 * np.outer(np.sin(u_vals), np.sin(v_vals))
            z_sphere = 1.0 * np.outer(np.ones_like(u_vals), np.cos(v_vals))
            if plot_sphere:
                ax.plot_surface(x_sphere, y_sphere, z_sphere, color="yellow", alpha=1.0)

            all_x = []
            all_y = []
            all_z = []

            if plot_concentrations:
                all_x.append(np.array(self.concentrations.x_coords))
                all_y.append(np.array(self.concentrations.y_coords))
                all_z.append(np.array(self.concentrations.z_coords))

            if extent is None:
                for flux in self.fluxons.values():
                    all_x.append(flux.x_coords)
                    all_y.append(flux.y_coords)
                    all_z.append(flux.z_coords)
                all_x = np.concatenate(all_x) if all_x else np.array([0])
                all_y = np.concatenate(all_y) if all_y else np.array([0])
                all_z = np.concatenate(all_z) if all_z else np.array([0])
                min_x, max_x = np.min(all_x), np.max(all_x)
                min_y, max_y = np.min(all_y), np.max(all_y)
                min_z, max_z = np.min(all_z), np.max(all_z)
                max_range = max((max_x - min_x), (max_y - min_y), (max_z - min_z))
                mid_x = 0.5 * (max_x + min_x)
                mid_y = 0.5 * (max_y + min_y)
                mid_z = 0.5 * (max_z + min_z)
                ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
                ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
                ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)
            else:
                ax.set_xlim(-extent, extent)
                ax.set_ylim(-extent, extent)
                ax.set_zlim(-extent, extent)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Flux World CR {self.cr} Visualization")
            # plt.legend(loc="upper right")
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(self.out_dir, f"cr{self.cr}_f{self.nflx}_fluxlight_world.png"))
                zoom = 3
                ax.set_xlim(-zoom, zoom)
                ax.set_ylim(-zoom, zoom)
                ax.set_zlim(-zoom, zoom)
                plt.savefig(os.path.join(self.out_dir, f"cr{self.cr}_f{self.nflx}_fluxlight_world_zoom.png"))
                # plt.show(block=True)
                # Save six camera angle views for zoomed plot: positive and negative x, y, z
                for elev, azim, coord in [
                    (0, 180, "x"), (0, 0, "-x"),
                    (0, 90, "y"), (0, -90, "-y"),
                    (90, 0, "z"), (-90, 0, "-z")
                ]:
                    ax.view_init(elev=elev, azim=azim)
                    plt.savefig(os.path.join(self.out_dir, f"cr{self.cr}_f{self.nflx}_fluxlight_world_zoom_{coord}.png"))
                plt.close(fig)
            else:
                plt.show()
            print("Done with world plotting!")

        def generate_coordinate_grid(self, num_points_per_axis=20, padding=0.1):
            fx_min, fx_max = self.all_fx.min(), self.all_fx.max()
            fy_min, fy_max = self.all_fy.min(), self.all_fy.max()
            fz_min, fz_max = self.all_fz.min(), self.all_fz.max()
            dx = (fx_max - fx_min) * padding
            dy = (fy_max - fy_min) * padding
            dz = (fz_max - fz_min) * padding
            x = np.linspace(fx_min - dx, fx_max + dx, num_points_per_axis)
            y = np.linspace(fy_min - dy, fy_max + dy, num_points_per_axis)
            z = np.linspace(fz_min - dz, fz_max + dz, num_points_per_axis)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T * u.R_sun
            return coords

        def generate_slices(self, num_points=200, extent=1.5 * u.R_sun):
            lin = np.linspace(-extent.value, extent.value, num_points) * extent.unit
            planes = {
                "XY": {"fixed": "z", "xlabel": "X", "ylabel": "Y"},
                "XZ": {"fixed": "y", "xlabel": "X", "ylabel": "Z"},
                "YZ": {"fixed": "x", "xlabel": "Y", "ylabel": "Z"},
            }
            slices = {}
            for name, cfg in planes.items():
                fixed = cfg["fixed"]
                if fixed == "z":
                    X, Y = np.meshgrid(lin, lin, indexing="ij")
                    Z = np.zeros_like(X)
                    coords = np.stack([X, Y, Z], axis=-1)
                elif fixed == "y":
                    X, Z = np.meshgrid(lin, lin, indexing="ij")
                    Y = np.zeros_like(X)
                    coords = np.stack([X, Y, Z], axis=-1)
                elif fixed == "x":
                    Y, Z = np.meshgrid(lin, lin, indexing="ij")
                    X = np.zeros_like(Y)
                    coords = np.stack([X, Y, Z], axis=-1)
                coords_flat = coords.reshape(-1, 3)
                densities = self.compute_electron_density(coords_flat).reshape(num_points, num_points)
                densities = np.log10(densities.to_value())
                slices[name] = {
                    "plane": (X, Y) if fixed == "z" else (X, Z) if fixed == "y" else (Y, Z),
                    "density": densities,
                    "extent": [-extent.value, extent.value, -extent.value, extent.value],
                    "xlabel": cfg["xlabel"],
                    "ylabel": cfg["ylabel"],
                    "label": name,
                }
            return slices

        def plot_density(
            self, num_points=200, extent=1.5 * u.R_sun, title="Electron Density Visualization", save=False
        ):
            import matplotlib.pyplot as plt

            coords = self.generate_coordinate_grid(num_points_per_axis=20)
            densities = self.compute_electron_density(coords)
            coords = np.array(coords)
            densities = np.log10(np.array(densities))
            norm_densities = (densities - densities.min()) / (densities.max() - densities.min())
            norm_densities = np.clip(norm_densities, 0.05, 1.0)
            norm_densities[norm_densities <= np.nanmin(norm_densities)] = 0
            slice_data = self.generate_slices(num_points=num_points, extent=extent)
            fig = plt.figure(figsize=(14, 12))
            fig.suptitle(title)
            ax3d = fig.add_subplot(2, 2, 1, projection="3d")
            sc = ax3d.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], c=densities, cmap="viridis", alpha=norm_densities
            )
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            fig.colorbar(sc, ax=ax3d, label="Electron Density")
            slice_keys = ["XY", "XZ", "YZ"]
            for i, key in enumerate(slice_keys):
                ax = fig.add_subplot(2, 2, i + 2)
                data = slice_data[key]
                im = ax.imshow(
                    np.log10(data["density"]), extent=data["extent"], origin="lower", cmap="viridis", aspect="equal"
                )
                ax.set_xlabel(data["xlabel"])
                ax.set_ylabel(data["ylabel"])
                ax.set_title(f"{key} Plane at {data['label']} = 0")
                fig.colorbar(im, ax=ax, label="Electron Density")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                plt.savefig(os.path.join(self.out_dir, f"cr{self.cr}_f{self.nflx}_fluxlight_density.png"))
                plt.close(fig)
            else:
                plt.show()

        def compute_electron_density(self, coords=None, method="fluxel", **kwargs):
            if method == "vertex":
                return self.compute_electron_density_vertex(coords=coords, **kwargs)
            elif method == "fluxel":
                return self.compute_electron_density_fluxel(coords=coords, **kwargs)
            else:
                raise ValueError(f"Unknown density method '{method}'")

        @u.quantity_input(influence_length=u.R_sun)
        def compute_electron_density_vertex(self, coords=None, influence_length=1 * u.R_sun, scale=100):
            if coords is None and self.coords is not None:
                coords = self.coords
            elif coords is None:
                coords = self.generate_coordinate_grid()
            self.coords = coords
            if self.tree is None:
                points = []
                for flux in self.fluxons.values():
                    vertices = flux.get_vertices()
                    for pt in vertices:
                        points.append(pt)
                points = np.array(points)
                self.tree = cKDTree(points)
            distances, _ = self.tree.query(coords) * coords[0].unit
            r = np.linalg.norm(coords, axis=1)
            n0 = 4.2e8 * u.cm**-3
            base_density = n0 * 10 ** (4.32 * u.R_sun / r.to(u.R_sun))
            factor = scale * np.exp(-distances / influence_length)
            densities = base_density * (1 + factor)
            return densities

        @u.quantity_input(influence_length=u.R_sun)
        def compute_electron_density_fluxel(self, coords=None, influence_length=1 * u.R_sun, scale=100):
            if coords is None and self.coords is not None:
                coords = self.coords
            elif coords is None:
                coords = self.generate_coordinate_grid()
            self.coords = coords

            # Re-entry guard
            if getattr(self, "_currently_computing_density", False):
                print("Warning: compute_electron_density_fluxel is already running. Skipping re-entry.")
                return np.zeros(coords.shape[0]) * u.cm**-3
            self._currently_computing_density = True
            try:
                # print(f"Computing electron density using fluxel method for {len(coords)} coordinates...")

                # Build or use cached list of all fluxel segments (pairs of points)
                if self.segments is None:
                    print("Building fluxel segments for the first time...")
                    self.segments = []
                    for flux in self.fluxons.values():
                        verts = flux.get_vertices()
                        self.segments.extend([(verts[i], verts[i + 1]) for i in range(len(verts) - 1)])
                    self.segments = np.array(self.segments) * u.R_sun
                    print(f"Built {len(self.segments)} fluxel segments for distance computation.")
                else:
                    # print(f"Using cached {len(self.segments)} fluxel segments.")
                    pass
                segments = self.segments

                coords_val = coords.to_value(u.R_sun)

                # --- Efficient local search for nearest segments using KD-tree of vertices ---
                # Build or use cached vertex-to-segments dictionary and KD-tree
                if not hasattr(self, '_vertex_tree') or self._vertex_tree is None:
                    print("Building vertex KD-tree and vertex-to-segment lookup...")
                    all_vertices = []
                    vertex_to_segments = {}
                    for seg in self.segments.to_value(u.R_sun):
                        a = tuple(seg[0])
                        b = tuple(seg[1])
                        all_vertices.append(seg[0])
                        all_vertices.append(seg[1])
                        for v in (a, b):
                            vertex_to_segments.setdefault(v, []).append((seg[0], seg[1]))
                    all_vertices = np.array(all_vertices)
                    self._vertex_tree = cKDTree(all_vertices)
                    self._vertex_coords = all_vertices
                    self._vertex_to_segments = vertex_to_segments
                else:
                    vertex_to_segments = self._vertex_to_segments

                def point_to_segments_local(points, all_segments, vertex_tree, k=10):
                    # Query nearest vertices
                    distances, indices = vertex_tree.query(points, k=k)
                    if k == 1:
                        indices = indices[:, None]
                    min_dists = np.full(points.shape[0], np.inf)
                    for i, idxs in enumerate(indices):
                        segs = []
                        for idx in np.unique(idxs):
                            v = tuple(vertex_tree.data[idx])
                            segs.extend(vertex_to_segments.get(v, []))
                        if not segs:
                            continue
                        segs = np.array(segs)
                        seg_a = segs[:, 0]
                        seg_b = segs[:, 1]
                        ab = seg_b - seg_a
                        ab_dot = np.sum(ab**2, axis=1)
                        ap = points[i] - seg_a
                        t = np.clip(np.sum(ap * ab, axis=1) / ab_dot, 0, 1)
                        closest = seg_a + t[:, None] * ab
                        dists = np.linalg.norm(points[i] - closest, axis=1)
                        min_dists[i] = np.min(dists)
                    return min_dists

                # print("Computing distances from each point to nearest local fluxel segment...")
                distances = point_to_segments_local(coords_val, self.segments.to_value(u.R_sun), self._vertex_tree)
                distances = distances * u.R_sun
                # print("Distance computation complete. Calculating density profile...")

                r = np.linalg.norm(coords, axis=1)
                n0 = 4.2e8 * u.cm**-3
                base_density = n0 * 10 ** (4.32 * u.R_sun / r.to(u.R_sun))
                factor = scale * np.exp(-distances / influence_length)
                densities = base_density * (1 + factor)
                # print("Density computation complete.")
                return densities
            finally:
                self._currently_computing_density = False

        def compute_fluxon_id(self, coords=None):
            if coords is None and self.coords is not None:
                coords = self.coords
            elif coords is None:
                coords = self.generate_coordinate_grid()
            self.coords = coords
            if not hasattr(self, "id_tree") or self.id_tree is None:
                points = []
                vertex_fids = []
                for flux in self.fluxons.values():
                    vertices = flux.get_vertices()
                    for pt in vertices:
                        points.append(pt)
                        vertex_fids.append(flux.id)
                points = np.array(points)
                self.id_tree = cKDTree(points)
                self.vertex_fids = np.array(vertex_fids)
            distances, indices = self.id_tree.query(coords)
            fids = self.vertex_fids[indices]
            return fids

        def compute_cross_sectional_areas(self, method="halfplane", k_neighbors=6, smooth=True):
            """
            Estimate the perpendicular cross-sectional area at each vertex using either
            a convex hull method or a half-plane (Voronoi) method.

            Parameters
            ----------
            method : str, optional
                Options are:
                'convex'    : Use the convex hull of midpoints of the vertex and its neighbors.
                'halfplane' : Compute the intersection of half-planes defined by the perpendicular bisectors.
                Default is 'halfplane'.
            k_neighbors : int, optional
                Number of nearest neighbors to use (default is 6).

            Returns
            -------
            areas_by_fluxon : dict
                A dictionary mapping each fluxon id to an array of estimated areas for each vertex.
            """
            import numpy as np
            from scipy.spatial import ConvexHull, cKDTree
            self.method = method

            if method == "file":
                # Read ground truth areas from an output file.
                # The file should have columns: fluxon id, x, y, z, r, theta, phi, A, fr
                if not hasattr(self, "flux_area_file") or not self.flux_area_file:
                    raise ValueError(
                        "flux_area_file not specified. Please set self.flux_area_file to the path of the ground truth data file."
                    )
                gt_data = np.loadtxt(self.flux_area_file)

                # Reassign keys in self.fluxons (created from the .flux file) to sequential IDs.
                # The .flux file produced high-numbered IDs, so we sort those keys and map them
                # to 0, 1, 2, ... in order.
                sorted_keys = sorted(self.fluxons.keys())
                mapping = {old: new for new, old in enumerate(sorted_keys)}
                new_fluxons = {}
                for old, flux in self.fluxons.items():
                    new_id = mapping[old]
                    flux.id = new_id
                    new_fluxons[new_id] = flux
                self.fluxons = new_fluxons

                # Now, use the area file's first column as the new fluxon IDs.
                # These are "almost sequential" already.
                area_ids = gt_data[:, 0].astype(int)
                unique_ids = np.unique(area_ids)

                # Conversion factor: solar radius in meters
                RS = 696340000.0

                # Group the area file data by its pseudo-sequential IDs,
                # converting r from m to R☉ and A from m² to R☉².
                areas_by_fluxon = {}
                self.radius_by_fluxon = {}
                for fid in unique_ids:
                    rows = gt_data[area_ids == fid]
                    areas_by_fluxon[fid] = rows[:, 7] / (RS**2)
                    self.radius_by_fluxon[fid] = rows[:, 4] / RS
                self.areas_by_fluxon = areas_by_fluxon

                # Update existing fluxon objects (now keyed by sequential IDs) or create dummy fluxons
                # for any IDs in the area file that are missing.
                found, lost = 0, 0
                for fid in unique_ids:
                    if fid in self.fluxons:
                        self.fluxons[fid].set_areas(areas_by_fluxon[fid], raw_radius=self.radius_by_fluxon[fid])
                        found += 1
                    else:
                        rows = gt_data[area_ids == fid]
                        # Convert x, y, z positions from m to solar radii
                        x = rows[:, 1] / RS
                        y = rows[:, 2] / RS
                        z = rows[:, 3] / RS
                        dummy_fluxon = Fluxon(fid, start_fc=-1, end_fc=-2, flux=np.nan, x_coords=x, y_coords=y, z_coords=z)
                        dummy_fluxon.set_areas(areas_by_fluxon[fid], raw_radius=self.radius_by_fluxon[fid])
                        dummy_fluxon.kind = "open"  # Force open so it's included in open-field plots.
                        self.fluxons[fid] = dummy_fluxon
                        lost += 1
                # print(f"Found = {found}, Created dummy fluxons = {lost}")
                return self.areas_by_fluxon

            def clip_polygon_by_halfplane(polygon, a, b, c):
                new_polygon = []
                n = len(polygon)
                for i in range(n):
                    curr = polygon[i]
                    nxt = polygon[(i + 1) % n]
                    curr_inside = a * curr[0] + b * curr[1] <= c
                    nxt_inside = a * nxt[0] + b * nxt[1] <= c
                    if curr_inside and nxt_inside:
                        new_polygon.append(nxt)
                    elif curr_inside and not nxt_inside:
                        d = a * (nxt[0] - curr[0]) + b * (nxt[1] - curr[1])
                        if d != 0:
                            t = (c - (a * curr[0] + b * curr[1])) / d
                            new_point = (curr[0] + t * (nxt[0] - curr[0]), curr[1] + t * (nxt[1] - curr[1]))
                            new_polygon.append(new_point)
                    elif not curr_inside and nxt_inside:
                        d = a * (nxt[0] - curr[0]) + b * (nxt[1] - curr[1])
                        if d != 0:
                            t = (c - (a * curr[0] + b * curr[1])) / d
                            new_point = (curr[0] + t * (nxt[0] - curr[0]), curr[1] + t * (nxt[1] - curr[1]))
                            new_polygon.append(new_point)
                        new_polygon.append(nxt)
                return new_polygon

            def polygon_area(polygon):
                if len(polygon) < 3:
                    return np.nan
                x = np.array([p[0] for p in polygon])
                y = np.array([p[1] for p in polygon])
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # Collect all vertices and store fluxon references as (fluxon_id, vertex_index)
            all_points = []
            flux_refs = []
            for flux in self.fluxons.values():
                vertices = flux.get_vertices()
                for j, pt in enumerate(vertices):
                    all_points.append(pt)
                    flux_refs.append((flux.id, j))
            all_points = np.array(all_points)

            tree = cKDTree(all_points)

            self.areas_by_fluxon = {}
            self.radius_by_fluxon = {}
            for flux in self.fluxons.values():
                self.areas_by_fluxon[flux.id] = np.full(len(flux.x_coords), np.nan)
                self.radius_by_fluxon[flux.id] = self.get_fluxon(flux.id).get_radius()

            from tqdm import tqdm
            for idx, pt in enumerate(tqdm(all_points, desc=f"Computing areas with {method = }")):
                fid, v_idx = flux_refs[idx]
                flux_obj = self.get_fluxon(fid)
                pts = flux_obj.get_vertices()
                if len(pts) < 2:
                    continue
                if v_idx == 0:
                    tangent = pts[1] - pts[0]
                elif v_idx == len(pts) - 1:
                    tangent = pts[-1] - pts[-2]
                else:
                    tangent = pts[v_idx + 1] - pts[v_idx - 1]
                norm = np.linalg.norm(tangent)
                if norm == 0:
                    continue
                tangent = tangent / norm

                arbitrary = np.array([0, 0, 1])
                if np.allclose(np.abs(np.dot(tangent, arbitrary)), 1.0):
                    arbitrary = np.array([0, 1, 0])
                v1 = np.cross(tangent, arbitrary)
                if np.linalg.norm(v1) == 0:
                    continue
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(tangent, v1)

                distances, indices = tree.query(pt, k=k_neighbors + 1)
                neighbor_indices = indices[1:]
                neighbor_points = all_points[neighbor_indices]

                projected_neighbors = []
                projected_midpoints = []
                for n_pt in neighbor_points:
                    diff = n_pt - pt
                    proj = np.array([np.dot(diff, v1), np.dot(diff, v2)])
                    projected_neighbors.append(proj)
                    midpoint = diff / 2.0
                    projected_midpoints.append(np.array([np.dot(midpoint, v1), np.dot(midpoint, v2)]))
                projected_neighbors = np.array(projected_neighbors)
                projected_midpoints = np.array(projected_midpoints)

                if method == "convex":
                    if projected_midpoints.shape[0] < 3:
                        area = np.nan
                    else:
                        try:
                            hull = ConvexHull(projected_midpoints)
                            area = hull.volume
                        except Exception:
                            area = np.nan
                elif method == "halfplane":
                    L = np.max(np.linalg.norm(projected_neighbors, axis=1)) * 2
                    polygon = [(-L, -L), (L, -L), (L, L), (-L, L)]
                    for n in projected_neighbors:
                        a = n[0]
                        b = n[1]
                        c = 0.5 * (a * a + b * b)
                        polygon = clip_polygon_by_halfplane(polygon, a, b, c)
                        if len(polygon) < 3:
                            break
                    area = polygon_area(polygon)
                else:
                    area = np.nan

                self.areas_by_fluxon[fid][v_idx] = area

            for flux in self.fluxons.values():

                flux_obj = self.get_fluxon(flux.id)
                raw_area = self.areas_by_fluxon[flux.id]
                save_area = raw_area if smooth is None else self.smooth_areas(raw_area, smooth)
                flux_obj.set_areas(save_area)

            return self.areas_by_fluxon

        def smooth_areas(self, raw_area, smooth=True, method='gaussian'):
            """
            Smooth the cross-sectional area data.

            Parameters:
                raw_area (array-like): The raw area data to be smoothed.
                smooth (bool or tuple): If True, use default parameters; if a tuple is provided, use those values.
                method (str, optional): Smoothing method to use. Options are 'savgol' for Savitzky–Golay filtering (default)
                                        and 'gaussian' for Gaussian smoothing.

            Returns:
                np.ndarray: The smoothed area data.
            """
            # If Gaussian smoothing is requested
            if method == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                if smooth is True:
                    sigma = 2.5
                elif isinstance(smooth, (tuple, list)) and len(smooth) >= 1:
                    sigma = smooth[0]
                else:
                    return raw_area
                return gaussian_filter1d(raw_area, sigma=sigma)
            elif method != 'savgol':
                # If an unsupported method is provided, return raw_area unmodified.
                return raw_area

            # Otherwise, use Savitzky–Golay smoothing
            from scipy.signal import savgol_filter

            # Determine filter parameters based on the 'smooth' parameter
            if smooth is True:
                window_length = 7
                polyorder = 4
            elif isinstance(smooth, (tuple, list)) and len(smooth) == 2:
                window_length, polyorder = smooth
            else:
                return raw_area

            # Ensure window_length is an odd integer
            window_length = int(window_length)
            if window_length % 2 == 0:
                window_length += 1

            # Adjust window_length if the raw_area length is smaller
            if len(raw_area) < window_length:
                window_length = len(raw_area) if len(raw_area) % 2 == 1 else len(raw_area) - 1
                if window_length < 3:
                    return raw_area

            if polyorder >= window_length:
                polyorder = window_length - 1

            try:
                smoothed_area = savgol_filter(raw_area, window_length=window_length, polyorder=polyorder)
            except Exception as e:
                print(e)
                return raw_area
            return smoothed_area

        def plot_all_fluxon_area_methods(self, save=True):

            for method in ["file", "convex", "halfplane"]:
                self.plot_fluxon_areas(method=method)
                print(f"Saved {method}")

        def plot_fluxon_areas(self, save=True, method="halfplane"):
            fig, axarray = plt.subplots(4, 1, sharex="all", figsize=(8, 10))

            (ax, ax2, ax3, ax4)  = axarray.flatten()

            for axy in axarray.flatten():
                axy.set_yscale("log")
                axy.set_xscale("log")
                axy.set_xlim(10**-2, 21.5)

            if self.areas_by_fluxon is None or not self.method == method:
                self.areas_by_fluxon = None
                self.compute_cross_sectional_areas(smooth=True, method=method)

            cl_ind, open_ind = 0, 0
            cl_tot, open_tot = 0, 0
            zr = None
            iterable = self.fluxons.values()

            for fluxonn in iterable:
                if fluxonn.kind == "closed":
                    cl_tot += 1.25
                elif fluxonn.kind == "open":
                    open_tot += 1.5

            if zr is None:
                zr = np.linspace(10**-3, 21.5)
                ax.plot(zr, (zr) ** 2, ls="--", c="k", zorder=10000, lw=3, alpha=0.75)
                ax3.plot(zr, (zr) ** 2, ls="--", c="k", zorder=10000, lw=3, alpha=0.75)
                ax2.axhline(1, ls="--", c="k", zorder=10000, lw=3, alpha=0.75)
                ax4.axhline(1, ls="--", c="k", zorder=10000, lw=3, alpha=0.75)

            for fluxonn in iterable:
                if fluxonn.areas is not None and not fluxonn.disabled:
                    if fluxonn.kind == "closed":
                        cl_ind += 1
                        ax3.plot(fluxonn.radius - 1, fluxonn.areas, color=plt.cm.Reds_r((cl_ind) / cl_tot), alpha=0.7, zorder=0)
                        ax4.plot(fluxonn.radius - 1, fluxonn.fr, color=plt.cm.Reds_r((cl_ind) / cl_tot), alpha=0.7, zorder=0)
                    elif fluxonn.kind == "open":
                        open_ind += 1
                        ax.plot(fluxonn.radius - 1, fluxonn.areas, color=plt.cm.Greens_r((open_ind) / open_tot), alpha=0.7, zorder=1000)
                        ax2.plot(fluxonn.radius - 1, fluxonn.fr, color=plt.cm.Greens_r((open_ind) / open_tot), alpha=0.7, zorder=1000)

            fig.suptitle(f"Area determination method: {self.method}")
            ax.set_title(f"Cross-sectional Area: Open Fields")
            ax2.set_title(f"Expansion Factor: Open Fields")
            ax3.set_title(f"Cross-sectional Area: Closed Fields")
            ax4.set_title(f"Expansion Factor: Closed Fields")
            plt.tight_layout()
            if save:
                out = os.path.normpath(os.path.join(self.out_dir, "..", "fr", f"cr{self.cr}_f{self.nflx}_fluxlight_area_{self.method}.png"))
                plt.savefig(out)
                plt.close(fig)
            else:
                plt.show()

            plt.close(fig)

        def plot_all_area_methods(self, save=True):
            for lines in [True, False]:
                self.plot_all_open_area_methods(save=save, lines=lines)

        def plot_all_open_area_methods(self, save=True, lines=False):
            """
            Plot the open-field fluxon cross-sectional areas and expansion factors computed
            by all three methods ("file", "convex", "halfplane") on the same figure.
            Each method is plotted in a different color.
            """
            import matplotlib.pyplot as plt
            methods = ["convex", "halfplane", "file"]
            colors = ["green", "blue", "red"]
            color_mapping = {
                "red": {"dark": "darkred", "light": "lightcoral"},
                "green": {"dark": "darkgreen", "light": "lightgreen"},
                "blue": {"dark": "darkblue", "light": "skyblue"},
            }
            # Dictionaries to store (radius, area) and (radius, expansion factor) for each method.
            results_area = {method: [] for method in methods}
            results_fr = {method: [] for method in methods}

            # Compute and store data for each method.
            fluxy = 0
            for method in methods:
                self.compute_cross_sectional_areas(smooth=True, method=method)
                for flux in self.fluxons.values():
                    if flux.kind == "open" and flux.areas is not None and not flux.disabled:
                        fluxy +=1
                        results_area[method].append((flux.radius.copy(), flux.areas.copy()))
                        results_fr[method].append((flux.radius.copy(), flux.fr.copy()))
            # print(F"{fluxy = }")
            # Create a figure with two subplots: one for area, one for expansion factor.
            fig, (ax_area, ax_fr) = plt.subplots(1, 2, figsize=(12, 6), sharex="all", sharey="none")

            # For legend control (so each method is labeled only once)
            labels_added_area = {method: False for method in methods}
            labels_added_fr = {method: False for method in methods}

            # Define a common grid for height (flux.radius - 1)
            common_grid = np.logspace(np.log10(1e-2), np.log10(21.5), 100)

            # Plot individual fluxon curves and compute mean and std via interpolation
            for method, color in zip(methods, colors):
                # Plot individual curves for visual reference
                for r, area in results_area[method]:
                    label = method if not labels_added_area[method] else None
                    if lines:
                        ax_area.plot(r - 1, area, color=color_mapping[color]["light"], alpha=0.25, label=label)
                    labels_added_area[method] = True

                # Interpolate each fluxon's area onto the common grid
                interp_areas = []
                for r, area in results_area[method]:
                    # Interpolate fluxon area vs (radius - 1) onto the common grid
                    interp_area = np.interp(common_grid, r - 1, area, left=area[0], right=area[-1])
                    interp_areas.append(interp_area)
                interp_areas = np.array(interp_areas)

                # Compute log-space statistics for error bars

                mean_curve, std_curve = self.log_stats(interp_areas, sig=2)

                err_curve = None if lines else std_curve
                import matplotlib.colors as mcolors

                transparent_color = None if lines else mcolors.to_rgba(color_mapping[color]["light"], alpha=0.75)

                # Plot the mean curve with error bars
                ax_area.errorbar(
                    common_grid,
                    mean_curve,
                    yerr=err_curve,
                    lw=4,
                    ls="--",
                    color=color_mapping[color]["dark"],
                    ecolor=transparent_color,
                    label=f"{method} mean",
                    zorder=100000,
                )
                ax_area.errorbar(
                    common_grid, mean_curve, yerr=err_curve, lw=5, ls="-", color=f"white", zorder=99999
                )

            # --- Plot expansion factor data ---
            # First, plot individual expansion factor curves for visual reference
            for method, color in zip(methods, colors):
                for r, fr in results_fr[method]:
                    label = method if not labels_added_fr[method] else None
                    if lines:
                        ax_fr.plot(r - 1, fr, color=color_mapping[color]["light"], alpha=0.25, label=label)
                    labels_added_fr[method] = True

            # Define a common grid for interpolation (same as used for areas)
            # (common_grid already defined above)
            # Now, compute and plot the errorbar summary for expansion factor data
            for method, color in zip(methods, colors):
                interp_fr = []
                for r, fr in results_fr[method]:
                    # Interpolate each fluxon's expansion factor (with x-axis given by r - 1) onto the common grid
                    interp_val = np.interp(common_grid, r - 1, fr, left=fr[0], right=fr[-1])
                    interp_fr.append(interp_val)
                interp_fr = np.array(interp_fr)
                mean_fr, std_fr = self.log_stats(interp_fr)

                err_curve2 = None if lines else np.abs(std_fr)
                transparent_color = None if lines else mcolors.to_rgba(color_mapping[color]["light"], alpha=0.75)

                ax_fr.errorbar(
                    common_grid,
                    mean_fr,
                    yerr=err_curve2,
                    fmt="--",
                    lw=4,
                    color=color_mapping[color]["dark"],
                    ecolor=transparent_color,
                    label=f"{method} mean",
                    zorder=100000,
                )
                ax_fr.errorbar(
                    common_grid,
                    mean_fr,
                    # yerr=std_fr,
                    fmt="-",
                    lw=5,
                    color="white",
                    # label=f"{method} mean",
                    zorder=99999,
                )

            ax_area.set_xlabel("Height above Photosphere [$R_\\odot$] - 1")
            ax_area.set_ylabel("Cross-sectional Area [R$_\\odot^2$]")
            ax_area.set_title("Cross-sectional Area (Open Fields)")
            ax_area.set_yscale("log")
            ax_area.set_xscale("log")

            ax_fr.set_xlabel("Height above Photosphere [R$_\\odot$] - 1")
            ax_fr.set_ylabel("Expansion Factor")
            ax_fr.set_title("Expansion Factor (Open Fields)")
            ax_fr.set_yscale("linear")
            ax_fr.set_xscale("log")

            ax_fr.set_xlim(10**-2, 21.5)
            ax_fr.set_ylim(-1, 16)

            zr = np.linspace(10**-2, 21.5)
            ax_area.plot(zr, (zr) ** 2, ls="--", c="k", zorder=1000000, lw=3, alpha=0.75, label="$R^2$")
            ax_area.plot(zr, (zr) ** 2, ls="-", c="w", zorder=999999, lw=4, alpha=0.75)
            ax_fr.axhline(1, ls="--", c="k", zorder=100000, lw=3, alpha=0.75, label="Unity")
            ax_fr.axhline(1, ls="-", c="w", zorder=99999, lw=4, alpha=0.75)
            # ax4.axhline(1, ls="--", c="k", zorder=10000, lw=3, alpha=0.75)

            fig.suptitle(f"Fluxon Expansion for CR {self.cr}")

            ax_area.legend()
            ax_fr.legend()
            plt.tight_layout()
            if save:
                out = os.path.normpath(os.path.join(self.out_dir, "..", "fr", f"cr{self.cr}_f{self.nflx}_open_area_all_methods_{lines=}.png"))
                plt.savefig(out)
                print(f"\nSaved figure to {out} !\n")
                plt.close(fig)
            else:
                plt.show()

        def log_stats(self, arr, sig=1):
            log_interp_arr = np.log(arr)
            mean_log_curve = np.nanmean(log_interp_arr, axis=0)
            std_log_curve = np.nanstd(log_interp_arr, axis=0)
            mean_curve = np.exp(mean_log_curve)
            lower_bound = np.exp(mean_log_curve - sig*std_log_curve)
            upper_bound = np.exp(mean_log_curve + sig*std_log_curve)
            std_curve = [mean_curve - lower_bound, upper_bound - mean_curve]
            return mean_curve, std_curve

        @u.quantity_input(displacement=u.R_sun)
        def plot_fluxon_id(
            self,
            num_points=600,
            extent=3.0 * u.R_sun,
            title="Fluxon ID Visualization",
            save=False,
            displacement=0.25 * u.R_sun,
        ):
            coords3d = self.generate_coordinate_grid(num_points_per_axis=num_points//20)
            fluxon_ids_3d = self.compute_fluxon_id(coords3d)
            lin = np.linspace(-extent.value, extent.value, num_points) * extent.unit
            slices = {}
            planes = {
                "XY": {"fixed": "z", "xlabel": "X", "ylabel": "Y"},
                "XZ": {"fixed": "y", "xlabel": "X", "ylabel": "Z"},
                "YZ": {"fixed": "x", "xlabel": "Y", "ylabel": "Z"},
            }
            for name, cfg in planes.items():
                fixed = cfg["fixed"]
                if fixed == "z":
                    X, Y = np.meshgrid(lin, lin, indexing="ij")
                    Z = np.ones_like(X) + displacement
                    coords_plane = np.stack([X, Y, Z], axis=-1)
                elif fixed == "y":
                    X, Z = np.meshgrid(lin, lin, indexing="ij")
                    Y = np.ones_like(X) + displacement
                    coords_plane = np.stack([X, Y, Z], axis=-1)
                elif fixed == "x":
                    Y, Z = np.meshgrid(lin, lin, indexing="ij")
                    X = np.ones_like(Y) + displacement
                    coords_plane = np.stack([X, Y, Z], axis=-1)
                coords_flat = coords_plane.reshape(-1, 3)
                fid_values = self.compute_fluxon_id(coords_flat).reshape(num_points, num_points)
                slices[name] = {
                    "plane": (X, Y) if fixed == "z" else (X, Z) if fixed == "y" else (Y, Z),
                    "fid": fid_values,
                    "extent": [-extent.value, extent.value, -extent.value, extent.value],
                    "xlabel": cfg["xlabel"],
                    "ylabel": cfg["ylabel"],
                    "label": fixed,
                }
            fig = plt.figure(figsize=(14, 12))
            fig.suptitle(title)
            ax3d = fig.add_subplot(2, 2, 1, projection="3d")
            sc = ax3d.scatter(
                coords3d[:, 0].value, coords3d[:, 1].value, coords3d[:, 2].value, c=fluxon_ids_3d, cmap="prism"
            )
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            fig.colorbar(sc, ax=ax3d, label="Fluxon ID")
            slice_keys = ["XY", "XZ", "YZ"]
            for i, key in enumerate(slice_keys):
                ax = fig.add_subplot(2, 2, i + 2)
                data = slices[key]
                im = ax.imshow(data["fid"], extent=data["extent"], origin="lower", cmap="prism", aspect="equal")
                ax.set_xlabel(data["xlabel"])
                ax.set_ylabel(data["ylabel"])
                ax.set_title(f"{key} Plane at {data['label']} = {displacement:0.3f}")
                fig.colorbar(im, ax=ax, label="Fluxon ID")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                plt.savefig(os.path.join(self.out_dir, f"cr{self.cr}_f{self.nflx}_fluxlight_ID.png"))
                plt.close(fig)
            else:
                plt.show()

        class FluxConcentrations:
            def __init__(self):
                self.ids = []  # Concentration ID Number
                self.x_coords = []  # Concentration X coordinate
                self.y_coords = []  # Concentration Y coordinate
                self.z_coords = []  # Concentration Z coordinate
                self.flux_values = []  # Flux Value

            def __str__(self):
                key_color = "\033[94m"
                reset_color = "\033[0m"
                return (
                    f"{key_color}FluxConcentration{reset_color} with {len(self.ids)} entries:\n"
                    f"{key_color}IDs{reset_color}: Length={len(self.ids)}, First={self.ids[:3]}, Last={self.ids[-3:]}\n"
                    f"{key_color}Flux values{reset_color}: Length={len(self.flux_values)}, First={self.flux_values[:3]}, Last={self.flux_values[-3:]}"
                )

        def add_flux_concentration(self, id, x, y, z, flux):
            self.concentrations.ids.append(id)
            self.concentrations.x_coords.append(x)
            self.concentrations.y_coords.append(y)
            self.concentrations.z_coords.append(z)
            self.concentrations.flux_values.append(flux)

        def add_fluxon(self, id, start_fc, end_fc, flux, start_pos, end_pos, x, y, z):
            new_fluxon = Fluxon(
                id,
                start_fc,
                end_fc,
                flux,
                [start_pos[0]] + x + [end_pos[0]],
                [start_pos[1]] + y + [end_pos[1]],
                [start_pos[2]] + z + [end_pos[2]],
            )
            self.fluxons[new_fluxon.id] = new_fluxon

        def get_fluxon(self, fid):
            return self.fluxons.get(fid, None)

    if os.path.isdir(filename):
        import re
        files = os.listdir(filename)
        largest_file = max(
            files,
            key=lambda f: int(re.search(r'_s(\d+)_', f).group(1)) if re.search(r'_s(\d+)_', f) else -1
        )
        filename = os.path.join(filename, largest_file)

    # Initialize the FluxWorld object
    flux_world = FluxWorld(filename)

    print("Parsing FLUX world data into memory...")
    # Reading data from the file
    with open(filename, "r") as file:
        line_ids = []
        line_start_fc = []
        line_end_fc = []
        line_flux_values = []
        line_start_pos = []
        line_end_pos = []

        vertex_line_ids = []
        vertex_ids = []
        vertex_positions = []
        vertex_x_coords = []
        vertex_y_coords = []
        vertex_z_coords = []

        for line in file:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "NEW":
                flux_world.add_flux_concentration(
                    int(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5])
                )
            elif tokens[0] == "LINE":
                if int(tokens[1]) > 0:
                    line_ids.append(int(tokens[1]))
                    line_start_fc.append(int(tokens[4]))
                    line_end_fc.append(int(tokens[5]))
                    line_flux_values.append(float(tokens[6]))
                    line_start_pos.append([float(tokens[7]), float(tokens[8]), float(tokens[9])])
                    line_end_pos.append([float(tokens[10]), float(tokens[11]), float(tokens[12])])
            elif tokens[0] == "VERTEX":
                vertex_line_ids.append(int(tokens[1]))
                vertex_ids.append(int(tokens[2]))
                vertex_positions.append(int(tokens[3]))
                vertex_x_coords.append(float(tokens[4]))
                vertex_y_coords.append(float(tokens[5]))
                vertex_z_coords.append(float(tokens[6]))
            elif "VNEIGHBOR" in tokens[0]:
                break

    print("Finished parsing file lines. Creating fluxon objects...")

    line_ids = np.array(line_ids)
    line_start_fc = np.array(line_start_fc)
    line_end_fc = np.array(line_end_fc)
    line_flux_values = np.array(line_flux_values)
    line_start_pos = np.array(line_start_pos)
    line_end_pos = np.array(line_end_pos)

    vertex_line_ids = np.array(vertex_line_ids)
    vertex_ids = np.array(vertex_ids)
    vertex_positions = np.array(vertex_positions)
    vertex_x_coords = np.array(vertex_x_coords)
    vertex_y_coords = np.array(vertex_y_coords)
    vertex_z_coords = np.array(vertex_z_coords)

    # Create fluxon objects by parsing line and vertex data
    for lid in line_ids:
        line_indices = np.where(line_ids == lid)[0]
        vertex_indices = np.where(vertex_line_ids == lid)[0]
        flux_world.add_fluxon(
            lid,
            line_start_fc[line_indices][0],
            line_end_fc[line_indices][0],
            line_flux_values[line_indices][0],
            line_start_pos[line_indices][0].tolist(),
            line_end_pos[line_indices][0].tolist(),
            vertex_x_coords[vertex_indices].tolist(),
            vertex_y_coords[vertex_indices].tolist(),
            vertex_z_coords[vertex_indices].tolist(),
        )
    print("Fluxons constructed.")
    print("Flux world loaded:", os.path.basename(filename))
    print(flux_world)
    flux_world.all_fx = np.concatenate([flux.x_coords for flux in flux_world.fluxons.values()])
    flux_world.all_fy = np.concatenate([flux.y_coords for flux in flux_world.fluxons.values()])
    flux_world.all_fz = np.concatenate([flux.z_coords for flux in flux_world.fluxons.values()])
    print("Computed fluxon bounding boxes.")
    # print("Fluxon bounding box (X):", flux_world.all_fx.min(), flux_world.all_fx.max())
    # print("Fluxon bounding box (Y):", flux_world.all_fy.min(), flux_world.all_fy.max())
    # print("Fluxon bounding box (Z):", flux_world.all_fz.min(), flux_world.all_fz.max())

    return flux_world


if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        the_world_file = sys.argv[1]
    else:
        the_world_file = "/Users/cgilbert/vscode/fluxons/fluxpype/fluxpype/data/batches/fluxlight/data/cr2150/world/cr2150_f1000_hmi_relaxed_s300.flux"

    f_world = read_flux_world(the_world_file)
    f_world.plot_all(save=True)
