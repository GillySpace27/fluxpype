"""Every plot the FLUX pipeline defines, so the dashboard can show each one: rendered when its file exists, as a
placeholder when it does not.

Compiled on 2026-11-11 by reading the pipeline code (call sites in science/magnetogram2wind.pdl and the plotting
scripts), not by running it. ``pattern`` is a glob relative to ``<data_dir>/batches/<batch>/`` in which only
``{cr}`` and ``{nflux}`` are filled in. ``legacy`` marks plots whose call is commented out, unreachable or manual
only: they are listed so nothing defined is hidden, but the current pipeline does not make them.
"""

from typing import Any, Dict, List

_PDL = "fluxpype/science/magnetogram2wind.pdl"
_MASTER = "fluxpype/science/map_fluxon_flow_parallel_master.pm"


ALLOWED_PLACEHOLDERS = frozenset(
    "python plotting science data_dir batch batch_dir cr nflux adapt inst red".split()
)  # plus in_<need> for each key of an entry's ``needs``


def _p(
    id,
    stage,
    title,
    pattern,
    writer,
    summary,
    call=None,
    gate=None,
    legacy=False,
    post_hoc=False,
    reason=None,
    command=None,
    needs=None,
    group=None,
    timeout_s=900,
    overwrite_risk=None,
    command_ref=None,
) -> Dict[str, Any]:
    """``command`` is an argv template (``{name}`` placeholders, see ALLOWED_PLACEHOLDERS) that remakes the plot
    from files already on disk; ``needs`` maps a name to a glob relative to data_dir that supplies ``{in_<name>}``.
    A post-hoc plot has a command; any other plot has a ``reason`` it cannot be run alone."""
    return {
        "id": id,
        "stage": stage,
        "title": title,
        "pattern": pattern,
        "writer": writer,
        "call": call,
        "gate": gate,
        "summary": summary,
        "legacy": legacy,
        "post_hoc": post_hoc,
        "post_hoc_reason": reason,
        "command": command,
        "command_ref": command_ref,
        "needs": needs or {},
        "group": group,
        "timeout_s": timeout_s,
        "overwrite_risk": overwrite_risk,
    }


PLOTS: List[Dict[str, Any]] = [
    # ---- made by the current pipeline
    _p(
        "fieldmap_topology",
        "fieldmap",
        "Magnetogram with fluxon footpoints",
        "imgs/footpoints/cr{cr}_f{nflux}_ou*_footpoints_topology.png",
        "fluxpype/plotting/plot_fieldmap.py:243",
        "Magnetogram (longitude 0 to 2 pi, sine latitude) with all fluxon roots (orange and teal by sign) and the open-field roots (red and blue). The name has no instrument or adapt tag.",
        call=f"{_PDL}:339",
        gate="Skipped when wind_only is set.",
    ),
    _p(
        "world_relaxed_narrow",
        "world_plots",
        "Relaxed world, narrow zoom",
        "imgs/world/0_narrow/cr{cr}_f{nflux}_relaxed_narrow.png",
        "fluxpype/plotting/plot_world.pm:91",
        "Gnuplot 3D render of the relaxed fluxon world in a cube of +-0.5 inner_fov solar radii (1.25 with the default 2.5).",
        call=f"{_PDL}:322",
        gate="Skipped when wind_only is set.",
    ),
    _p(
        "world_relaxed_middle",
        "world_plots",
        "Relaxed world, middle zoom",
        "imgs/world/1_middle/cr{cr}_f{nflux}_relaxed_middle.png",
        "fluxpype/plotting/plot_world.pm:95",
        "Gnuplot 3D render of the relaxed world in a cube of +-inner_fov solar radii.",
        call=f"{_PDL}:322",
        gate="Skipped when wind_only is set.",
    ),
    _p(
        "world_relaxed_wide",
        "world_plots",
        "Relaxed world, wide zoom",
        "imgs/world/2_wide/cr{cr}_f{nflux}_relaxed_wide.png",
        "fluxpype/plotting/plot_world.pm:99",
        "Gnuplot 3D render of the relaxed world in a cube of +-2 inner_fov solar radii.",
        call=f"{_PDL}:322",
        gate="Skipped when wind_only is set.",
    ),
    _p(
        "world_relaxed_full",
        "world_plots",
        "Relaxed world, full extent",
        "imgs/world/3_full/cr{cr}_f{nflux}_relaxed_full.png",
        "fluxpype/plotting/plot_world.pm:103",
        "Gnuplot 3D render of the relaxed world in a cube of +-215 solar radii (hard-coded).",
        call=f"{_PDL}:322",
        gate="Skipped when wind_only is set.",
    ),
    _p(
        "bmag_scatter",
        "wind_plots",
        "Fluxon B and area at the base and 21.5 Rsun",
        "imgs/bmag/cr{cr}_f{nflux}_radial_bmag.png",
        "fluxpype/plotting/plot_bmag.py:129",
        "Two panels over the magnetogram: field strength (top) and fluxon area (bottom) at 1.0 and 21.5 solar radii, sized by magnitude.",
        call=f"{_PDL}:421 (also run from fluxpype/science/map_fluxon_b.pm:116 in the wind stage)",
    ),
    _p(
        "bmag_all_profiles",
        "wind_plots",
        "Field strength and expansion factor against height",
        "imgs/bmag/all/sets/cr{cr}_all_f{nflux}/cr{cr}_f{nflux}_radial_bmag_all_*.png",
        "fluxpype/plotting/multiplot_bmag_all.py:220",
        "Log-log |B| and expansion factor of every open fluxon against height, with Zephyr curves and a sunspot-cycle panel. A copy is also written to imgs/bmag/all/bottom/. The file index is always 00 today.",
        call=f"{_PDL}:424",
        gate=None,
    ),
    _p(
        "wind_profiles_lines",
        "wind_plots",
        "Full velocity profiles, individual lines",
        "imgs/wind/full_velocity_profiles/lines_velocity_profile_comparison_cr{cr}.png",
        "fluxpype/plotting/plot_wind_solutions.py:170",
        "Velocity (km/s) against z = r/Rsun - 1 per fluxon for each wind method, a histogram at z = 21.5 and a Zephyr overlay. The name carries no fluxon count, so runs with different counts overwrite each other.",
        call=f"{_PDL}:445",
        gate="Loops over every rotation in the config; needs the per-method full_velocity .dat files.",
    ),
    _p(
        "wind_profiles_filled",
        "wind_plots",
        "Full velocity profiles, median and spread",
        "imgs/wind/full_velocity_profiles/filled_velocity_profile_comparison_cr{cr}.png",
        "fluxpype/plotting/plot_wind_solutions.py:170",
        "Median velocity with standard-deviation bars against z for each wind method, a histogram at z = 21.5 and a Zephyr overlay.",
        call=f"{_PDL}:445",
        gate="Made in the same pass as the lines plot, so it needs the same per-method full_velocity .dat files.",
    ),
    _p(
        "wsa_distances",
        "wind",
        "WSA distance to the coronal-hole boundary",
        "imgs/footpoints/distances_{cr}.png",
        "fluxpype/science/footpoint_distances_wsa.py:863",
        "Distance (degrees) to the nearest coronal-hole boundary against longitude and sine latitude, with raw and morphed contours and the closed-field overlay.",
        call=f"{_MASTER}:236",
        gate="Only for the wsa flow method.",
    ),
    _p(
        "wsa_distances_stamped",
        "wind",
        "WSA distance map, timestamped copy",
        "imgs/footpoints/t_distances_{cr}_*.png",
        "fluxpype/science/footpoint_distances_wsa.py:864",
        "The same figure with a unix timestamp in the name, so a new file appears on every run.",
        call=f"{_MASTER}:236",
        gate="Only for the wsa flow method.",
    ),
    _p(
        "wsa_expansion",
        "wind",
        "WSA predictors: distance, expansion, speed",
        "imgs/footpoints/expansion_{cr}.png",
        "fluxpype/science/footpoint_distances_wsa.py:538",
        "Three stacked maps: distance to the boundary, log10 expansion factor and WSA wind speed.",
        call=f"{_MASTER}:236",
        gate="Only for the wsa flow method.",
    ),
    _p(
        "wsa_morph_debug",
        "wind",
        "Coronal-hole morphology steps",
        "imgs/footpoints/morph_debug.png",
        "fluxpype/science/footpoint_distances_wsa.py:382",
        "A 2x4 panel of the boundary mask at each processing step. The name has no rotation, so each run overwrites it.",
        call=f"{_MASTER}:236",
        gate="Only for the wsa flow method; debug saving is switched on in the script.",
    ),
    _p(
        "tempest_velocity",
        "wind",
        "Tempest wind speed against height",
        "imgs/wind/tempest/tempest_cr{cr}.png",
        "fluxpype/science/tempest.py:1042 (cranmer: fluxpype/science/cranmer_wind.py:200)",
        "Prospero solar-wind speed curves (log-log) with transition-region and critical-point lines.",
        call=f"{_MASTER}:226 (tempest), :230 (cranmer)",
        gate="Only for the tempest or cranmer flow methods.",
    ),
    _p(
        "cranmer_bmag_zephyr",
        "wind",
        "FLUX against Zephyr field strength",
        "imgs/mag/cr{cr}.png",
        "fluxpype/science/cranmer_wind.py:275",
        "Interpolated FLUX B(r) per fluxon against Zephyr B(r).",
        call=f"{_MASTER}:230",
        gate="Only for the cranmer flow method, and only the first time (when its tempest file does not exist yet).",
    ),
    _p(
        "light_triplot",
        "fluxlight",
        "FLUXlight forward model (tB, pB, polarization fraction)",
        "imgs/light/cr{cr}_fluxlight_light_*_[0-9]*_[0-9]*_[0-9]*.png",
        "fluxpype/science/fluxlight.py:301",
        "Three panels in solar radii: total brightness, polarized brightness and polarization fraction. The name carries method, nz, crop radius and influence length; four crops (100, 50, 25, 10 Rsun) are made per run.",
        call=f"{_PDL}:476 (fluxlight.py:321-324)",
        gate="Only when the light setting is on.",
    ),
    # ---- defined in the code but not made by the current pipeline
    _p(
        "world_initial",
        "world_plots",
        "Initial (unrelaxed) world, four zoom levels",
        "imgs/world/[0-3]_*/cr{cr}_f{nflux}_initial_*.png",
        "fluxpype/plotting/plot_world.pm:91-103",
        "The same Gnuplot renders as the relaxed set, for the world before relaxation.",
        call="fluxpype/plotting/plot_world.pm:153-166 (commented out)",
        gate="The call is commented out.",
        legacy=True,
    ),
    _p(
        "magnetogram_raw",
        "magnetogram",
        "Raw magnetogram image",
        "../../magnetograms/CR{cr}_r*.png",
        "fluxpype/pipe_helper.py:1556",
        "Grayscale magnetogram with a 3 sigma stretch, plus a small copy, written beside the FITS file.",
        call="none (plot_raw_magnetogram has no caller)",
        gate="Never called.",
        legacy=True,
    ),
    _p(
        "footpoints_locations",
        "footpoints",
        "Footpoint locations over the magnetogram",
        "imgs/footpoints/cr{cr}_footpoints_r*_blank.png",
        "fluxpype/science/pfss_funcs.py:508-526",
        "A blank and an annotated magnetogram with the footpoint roots.",
        call="none (plot_fluxon_locations at pfss_funcs.py:411 is never called)",
        gate="Never called.",
        legacy=True,
    ),
    _p(
        "relax_movie",
        "relax",
        "Relaxation movie frames",
        "data/cr{cr}/world/flux-movie-*.png",
        "fluxpype/science/simple_relaxer.pm:251",
        "Gnuplot frames every 200 relaxation steps.",
        call="fluxpype/science/relax_pfss_world.pm:169",
        gate="Switched off: the display count passed in is 0.",
        legacy=True,
    ),
    _p(
        "fr_expansion",
        "wind_plots",
        "Fluxon expansion factors",
        "imgs/fr/cr{cr}_f{nflux}_radial_fr.png",
        "fluxpype/plotting/plot_fr.py:197",
        "Area and expansion factor against height, and an area map (read from the axis labels).",
        call=f"{_PDL}:433 (commented out)",
        gate="The call is commented out.",
        legacy=True,
    ),
    _p(
        "bmag_all_scatter",
        "wind_plots",
        "B and expansion multi-panel (plot_bmag_all)",
        "imgs/bmag/all/sets/cr{cr}_all_f{nflux}/cr{cr}_f{nflux}_radial_bmag_all_*.png",
        "fluxpype/plotting/plot_bmag_all.py:266",
        "Field strength and origin map at a chosen radius plus B and expansion against height. It writes to the same place as the live multi-panel.",
        call=f"{_PDL}:427 (commented out)",
        gate="The call is commented out.",
        legacy=True,
    ),
    _p(
        "bmag_fill",
        "wind_plots",
        "Geometric expansion multi-panel",
        "imgs/bmag/fill/sets/cr{cr}_fill_f{nflux}/cr{cr}_f{nflux}_radial_bmag_fill_*.png",
        "fluxpype/plotting/plot_bmag_fill.py:415",
        "Field-strength maps at low and high radius plus transverse, latitude and longitude displacement.",
        call=f"{_PDL}:430 (commented out)",
        gate="The call is commented out.",
        legacy=True,
    ),
    _p(
        "windmap",
        "wind_plots",
        "Wind maps (latitude, square and detailed variants)",
        "imgs/windmap/png_cr{cr}_f{nflux}_op*_radial_wind_*.png",
        "fluxpype/plotting/plot_wind_map_latitude_cycle.py:720 (also plot_wind_map_detailed.py:747, plot_wind_map_latitude.py:583, plot_wind_map_square.py:720)",
        "A histogram of wind speed at 21 Rsun, fluxon locations at the base and 21.5 Rsun, wind-speed dots, an expansion-ratio map and the sunspot phase. op is the number of open fluxons.",
        call=f"{_PDL}:436-442 (commented out)",
        gate="The calls are commented out.",
        legacy=True,
    ),
    _p(
        "world3d",
        "fluxlight",
        "3D flux world (matplotlib) with zoom and axis views",
        "imgs/world/cr{cr}_f{nflux}_fluxlight_world*.png",
        "fluxpype/science/open_world_python.py:451,456,468",
        "Fluxons by kind in 3D, plus a +-3 Rsun zoom and six axis views (+x, -x, +y, -y, +z, -z).",
        call=f"only open_world_python.py:1784 (run by hand); {_PDL}:447 and fluxlight.py:94 are commented out",
        gate="Manual run only.",
        legacy=True,
    ),
    _p(
        "world_density",
        "fluxlight",
        "Electron density slices",
        "imgs/world/cr{cr}_f{nflux}_fluxlight_density.png",
        "fluxpype/science/open_world_python.py:559",
        "A 3D scatter plus XY, XZ and YZ slices of the modelled electron density.",
        call="only open_world_python.py:1784",
        gate="Manual run only.",
        legacy=True,
    ),
    _p(
        "world_fluxon_id",
        "fluxlight",
        "Fluxon ID slices",
        "imgs/world/cr{cr}_f{nflux}_fluxlight_ID.png",
        "fluxpype/science/open_world_python.py:1612",
        "A 3D scatter and plane slices coloured by the owning fluxon.",
        call="only open_world_python.py:1784 (fluxlight.py:103 is commented out)",
        gate="Manual run only.",
        legacy=True,
    ),
    _p(
        "area_by_method",
        "fluxlight",
        "Fluxon area and expansion for one area method",
        "imgs/fr/cr{cr}_f{nflux}_fluxlight_area_*.png",
        "fluxpype/science/open_world_python.py:1364",
        "Four log-log panels: open and closed fluxon cross-sectional area and expansion factor against height.",
        call="only open_world_python.py:1784",
        gate="Manual run only.",
        legacy=True,
    ),
    _p(
        "open_area_all_methods",
        "fluxlight",
        "Open-field area and expansion, all area methods",
        "imgs/fr/cr{cr}_f{nflux}_open_area_all_methods_lines=*.png",
        "fluxpype/science/open_world_python.py:1533",
        "Area and expansion factor for the file, convex and halfplane methods on one figure (two files per run, with and without lines).",
        call="only open_world_python.py:1784",
        gate="Manual run only.",
        legacy=True,
    ),
    _p(
        "light_quad",
        "fluxlight",
        "FLUXlight quad plot",
        "imgs/light/cr{cr}_fluxlight_light_[a-z]*_[0-9].png",
        "fluxpype/science/fluxlight.py:235",
        "tB, pB, RHEF-filtered column density and polarization fraction in a 2x2 layout.",
        call="none (quadplot at fluxlight.py:185 is never called)",
        gate="Never called.",
        legacy=True,
    ),
    _p(
        "wsa_bkk_debug",
        "wind",
        "WSA predictor debug images (per fluxon)",
        "debug_fluxon_*.png",
        "fluxpype/science/gen_fluxon_wsaflow_bkk.pm:445,461",
        "Distance and expansion arrays as Gnuplot images, written to the working directory.",
        call="none (the module is not loaded anywhere)",
        gate="Never loaded.",
        legacy=True,
    ),
    _p(
        "footpoint_distance_variants",
        "wind",
        "Footpoint distance maps (older variants)",
        "data/cr{cr}/floc/distance_cr{cr}_f{nflux}_ou*_footpoints.png",
        "fluxpype/science/footpoint_distances.py:384 (and _2, _wsa_bkk, _wsa_bkk2)",
        "Footpoint topology over a distance-to-boundary map.",
        call="none (only footpoint_distances_wsa.py is called)",
        gate="Never called.",
        legacy=True,
    ),
    _p(
        "tempest_mag",
        "wind",
        "FLUX against Zephyr field (tempest.py blocks)",
        "imgs/mag2/cr{cr}.png",
        "fluxpype/science/tempest.py:1122,1261",
        "B(r) per fluxon against Zephyr B(r). Live only in cranmer_wind.py.",
        call="inside `doplot and False` blocks",
        gate="Unreachable.",
        legacy=True,
    ),
    _p(
        "fss_theta_speed",
        "wind_plots",
        "Speed against angle (plot_angles, plot_flux_mag)",
        "fluxon-data/fss_theta_speed.png",
        "fluxpype/plotting/plot_angles.py:67",
        "Wind speed against angle; written to a relative path outside any batch.",
        call="none (fluxpype/science/get_wind.pm:259 is commented out)",
        gate="Never called.",
        legacy=True,
    ),
]


# ---- which plots can be remade alone (compiled by reading the code; no command has been run)
_W = "batches/{batch}/data/cr{cr}/wind"
_F = "batches/{batch}/data/cr{cr}/floc"
_WR = "batches/{batch}/data/cr{cr}/world/cr{cr}_f{nflux}_{inst}_relaxed_s*.flux"
_C = ["--cr", "{cr}", "--nwant", "{nflux}", "--adapt", "{adapt}"]
_NAME_RISK = "No fluxon count or adapt in the figure or its input name; another run of this rotation in this batch writes the same file."
_NO_STANDALONE = (
    "Drawn by Gnuplot inside the Perl pipeline from the in-memory world (magnetogram2wind.pdl:322, "
    "plot_world.pm:91-103); no standalone command."
)
_WSA = (
    "footpoint_distances_wsa.py is a wind-computation step: it also writes the distance/expansion CSVs the wind "
    "stage reads back (footpoint_distances_wsa.py:757-801); rerunning it rewrites science inputs."
)
_NO_CALLER = "A library function with no caller and no command line."
_INSIDE = "Inside Perl, or code that is never loaded or never reached."
_OPEN_WORLD = dict(
    group="open_world",
    timeout_s=3600,
    command=["{python}", "{science}/open_world_python.py", "{in_world}"],
    needs={"world": _WR, "fr": f"{_W}/cr{{cr}}_f{{nflux}}_radial_fr.dat"},
    command_ref="fluxpype/science/open_world_python.py:1774-1784",
)
_WIND_SOL = dict(
    group="wind_solutions",
    command=["{python}", "{plotting}/plot_wind_solutions.py", "{batch_dir}/data/cr{cr}/wind"],
    needs={"fv": f"{_W}/full_velocity_profiles/results_*_full_velocity.dat"},
    command_ref=f"{_PDL}:445",
    overwrite_risk=_NAME_RISK,
)
_HOC: Dict[str, Dict[str, Any]] = {
    "fieldmap_topology": dict(
        command=[
            "{python}",
            "{plotting}/plot_fieldmap.py",
            "--cr",
            "{cr}",
            "--open",
            "{in_floc_open}",
            "--closed",
            "{in_floc_closed}",
            "--nwant",
            "{nflux}",
            "--adapt",
            "{adapt}",
        ],
        needs={
            "floc_open": f"{_F}/floc_open_cr{{cr}}_r{{red}}_f{{nflux}}_{{inst}}.dat",
            "floc_closed": f"{_F}/floc_closed_cr{{cr}}_r{{red}}_f{{nflux}}_{{inst}}.dat",
            "floc_all": f"{_F}/floc_cr{{cr}}_r{{red}}_f{{nflux}}_{{inst}}.dat",
            "mag": "magnetograms/CR{cr}_r{red}_{inst}.fits",
        },
        command_ref=f"{_PDL}:334-339",
    ),
    "bmag_scatter": dict(
        command=["{python}", "{plotting}/plot_bmag.py", "--file", "{in_bmag}", *_C],
        needs={"bmag": f"{_W}/cr{{cr}}_f{{nflux}}_radial_bmag.dat"},
        command_ref=f"{_PDL}:421",
    ),
    "bmag_all_profiles": dict(
        command=["{python}", "{plotting}/multiplot_bmag_all.py", "--file", "{in_bmag_all}", *_C],
        needs={"bmag_all": f"{_W}/cr{{cr}}_f{{nflux}}_radial_bmag_all.dat"},
        command_ref=f"{_PDL}:424",
    ),
    "wind_profiles_lines": _WIND_SOL,
    "wind_profiles_filled": _WIND_SOL,
    "light_triplot": dict(
        timeout_s=3600,
        command=["{python}", "{science}/fluxlight.py", "{in_world}"],
        needs={"world": _WR},
        command_ref=f"{_PDL}:360,476",
        overwrite_risk=_NAME_RISK.replace("figure or its input name", "figure name"),
    ),
    "fr_expansion": dict(
        command=["{python}", "{plotting}/plot_fr.py", "--file", "{in_fr}", *_C],
        needs={"fr": f"{_W}/cr{{cr}}_f{{nflux}}_radial_fr.dat"},
        command_ref=f"{_PDL}:433 (commented out)",
    ),
    "bmag_fill": dict(  # runtime unverified; also writes an .avi via cv2
        command=["{python}", "{plotting}/plot_bmag_fill.py", "--file", "{in_bmag_all}", *_C],
        needs={"bmag_all": f"{_W}/cr{{cr}}_f{{nflux}}_radial_bmag_all.dat"},
        command_ref=f"{_PDL}:430 (commented out)",
    ),
    **{
        i: _OPEN_WORLD
        for i in ("world3d", "world_density", "world_fluxon_id", "area_by_method", "open_area_all_methods")
    },
}
_NOT_HOC: Dict[str, Any] = {
    **{i: _NO_STANDALONE for i in ("world_relaxed_narrow", "world_relaxed_middle", "world_relaxed_wide")},
    "world_relaxed_full": _NO_STANDALONE,
    "world_initial": _NO_STANDALONE,
    **{i: _WSA for i in ("wsa_distances", "wsa_distances_stamped", "wsa_expansion", "wsa_morph_debug")},
    "tempest_velocity": "Written by the wind solver, which also writes wind data (tempest.py:1126,1180; cranmer_wind.py:279,320).",
    "cranmer_bmag_zephyr": "Written by the wind solver, which also writes wind data (tempest.py:1126,1180; cranmer_wind.py:279,320).",
    "bmag_all_scatter": "Same file name and folder as bmag_all_profiles, so it would replace the live plot; also writes an .avi.",
    "windmap": "Four variants, none verified: hard-coded 'sequential' batch; PDL form uses --resave_from_dat which exits before plotting.",
    **{i: _NO_CALLER for i in ("magnetogram_raw", "footpoints_locations", "light_quad")},
    **{i: _INSIDE for i in ("relax_movie", "wsa_bkk_debug", "tempest_mag")},
    **{
        i: "Not verified: never called; writes outside the data folder."
        for i in ("footpoint_distance_variants", "fss_theta_speed")
    },
}
for _e in PLOTS:
    if _e["id"] in _HOC:
        _e.update(_HOC[_e["id"]], post_hoc=True)
    else:
        _e["post_hoc_reason"] = _NOT_HOC[_e["id"]]
