# plot_final_network_maps.py
#
# One clean map per scenario (analysis/model_results/*_model_checkpoint.dill),
# showing the final year (2050) of the network -- i.e. the solution the
# optimization actually settled on, not a year-by-year animation. Builds on
# developed_plots.py (add_nodes, the booster-icon helper) but differs from
# it in three ways:
#
#   1. Built pipelines are colored by diameter, not by pressure/temperature
#      gradient. developed_plots.add_pipes() colors each built pipe along
#      its length by its simulated pressure and temperature drop -- useful
#      for inspecting *why* a booster was needed, but this figure only
#      needs to answer "was this pipe built, how big is it, is it insulated,
#      does it have a booster": add_network() below draws every candidate
#      pipe (gray, full topology) and every built pipe in a single flat
#      color per pipe, on a fixed diameter colorbar (6-42 inch, same scale
#      on every plot so pipe sizes are comparable across scenarios), with
#      the insulated-pipe gold highlight and booster icon kept.
#
#   2. The full candidate topology IS shown (unlike the previous version of
#      this script, which suppressed it for a decluttered "final network
#      only" look) -- every candidate pipeline the optimizer could have
#      built is drawn in gray behind the built ones, same as
#      developed_plots.add_pipes()'s background pass.
#
#   3. Plain basemap instead of the SRTM relief raster. developed_plots.py's
#      draw_relief() needs merged_srtm.tif, which isn't checked into the
#      repo (.gitignore'd -- it's a multi-MB DEM). Rather than depend on a
#      local, personal copy (romans_playground/merged_srtm_roman.tif is in a
#      different CRS besides), draw_basemap() below renders a simple country
#      map from the boundary GeoJSONs already in data/ (Spain, Portugal,
#      Andorra, France).
#
# Each output image is labeled with the scenario code (top-left) in place of
# developed_plots.py's small year badge, since year is now fixed at 2050 for
# every plot here.
#
# Usage (from the repo root):
#   python -m analysis.plot_final_network_maps

import sys
from collections import defaultdict
from pathlib import Path

import dill
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtrans
import numpy as np
import pyomo.environ as pyo
from matplotlib.collections import LineCollection
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iberian_co2_network.developed_plots import (  # noqa: E402
    add_nodes, booster_patch, _offset_linestring, _LANE_OFFSET_DEG,
)

CHECKPOINT_DIR = Path("analysis/model_results")
OUT_DIR = Path("analysis/results_plot_final")
YEAR = 2050
SCENARIO = "base_utilization"  # EUS: the base-case utilization scenario used throughout

# Map extent (lon/lat), fixed across every scenario so the plots are directly
# comparable; wide enough to frame mainland Iberia + Balearic Islands with a
# small margin (checked against every checkpoint's DATA["coords"] range).
EXTENT = (-10.3, 4.9, 35.7, 44.1)  # (lon_min, lon_max, lat_min, lat_max)

LAND_COLOR = "#EDE6D6"
BORDER_COLOR = "#6E6558"
REGION_BORDER_COLOR = "#C9C1B2"
SEA_COLOR = "#BCD8FF"
BASEMAP_COUNTRIES = ["spain", "portugal", "andorra", "france"]

CANDIDATE_COLOR = "#808080"
INSULATION_COLOR = "#E03131"

# Diameter color scale: fixed 6-42 inch range (the model's full candidate
# diameter set, DATA["D"]/m.D) so a given diameter maps to the same color on
# every scenario's plot -- pipe sizes stay comparable across figures, same
# principle as the pressure/temperature colorbars this replaces.
DIAMETER_MIN, DIAMETER_MAX = 6, 42
_diameter_cmap = plt.cm.viridis
_diameter_norm = mpl.colors.Normalize(vmin=DIAMETER_MIN, vmax=DIAMETER_MAX)


def _diameter_color(diam_inch):
    return _diameter_cmap(_diameter_norm(float(diam_inch)))


def draw_basemap(ax):
    """Plain country-boundary map (no relief/topology): land fill, national
    borders only, flat sea background -- see module docstring for why this
    replaces developed_plots.draw_relief()'s SRTM raster here.

    data/spain.json and data/portugal.json are collections of regional
    polygons (Spain's autonomous communities, Portugal's districts), not a
    single national outline. Drawn as three separate passes so the fill,
    the (paler, thinner) regional boundaries, and the (darker, thicker)
    national border -- the dissolved union of each country's regions --
    are each styled independently, with the national border on top."""
    ax.set_facecolor(SEA_COLOR)
    for country in BASEMAP_COUNTRIES:
        gdf = gpd.read_file(Path("data") / f"{country}.json")
        gdf.plot(ax=ax, facecolor=LAND_COLOR, edgecolor="none", zorder=0)
        gdf.boundary.plot(ax=ax, color=REGION_BORDER_COLOR, linewidth=0.4, zorder=0.3)
        gdf.dissolve().boundary.plot(ax=ax, color=BORDER_COLOR, linewidth=1.0, zorder=0.5)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def draw_candidate_topology(ax, DATA):
    """Every candidate pipeline (built or not), in gray -- the full graph
    the optimizer could have chosen from. Same as the background pass in
    developed_plots.add_pipes()."""
    for p, geom in DATA["pipe_geom"].items():
        stage = DATA["stage_p"][p]
        ls = "-" if stage == "First" else (0, (6, 10))
        x, y = geom.xy
        ax.plot(x, y, color=CANDIDATE_COLOR, linewidth=0.6, alpha=0.4, linestyle=ls, zorder=2)


def add_network(ax, DATA, model, year, w):
    """Built pipelines only: a single flat color (no pressure/temperature
    gradient), insulated pipes highlighted gold, booster stations marked,
    diameter labeled. The candidate background is drawn separately by
    draw_candidate_topology()."""
    m = model
    t_match = next(t for t in m.T if int(t) == year)

    def _is_active_pipe(p):
        if p in m.P_on:
            act = pyo.value(m.act_on[p, t_match, w]) > 0.5
            flow = any(pyo.value(m.q_on[p, tt, w]) > 0.1 for tt in m.T)
        else:
            act = (pyo.value(m.act_off[p, t_match, w]) +
                   (pyo.value(m.b_ship[p, t_match, w]) if p in m.P_off else 0.0)) > 0.5
            flow = any(pyo.value(m.q_off[p, tt, w]) > 0.1 for tt in m.T)
        return act and flow

    def _route_key(p):
        return frozenset((DATA["start"][p], DATA["end"][p]))

    active_by_route = defaultdict(list)
    for p in list(m.P_on) + list(m.P_off):
        if _is_active_pipe(p):
            active_by_route[_route_key(p)].append(p)

    node_points = [Point(xy) for xy in DATA["coords"].values()]
    label_centers = []

    for p in list(m.P_on) + list(m.P_off):
        is_on = (p in m.P_on)
        if not _is_active_pipe(p):
            continue

        u_selected = None
        if is_on:
            if p in m.P1_on:
                diam_selected, u_selected = next(
                    ((d, u) for d in m.D for u in m.U if pyo.value(m.b_diam_on_P1[p, d, u]) > 0.5),
                    (None, None))
            else:
                diam_selected, u_selected = next(
                    ((d, u) for d in m.D for u in m.U if pyo.value(m.b_diam_on_P2[p, d, u, w]) > 0.5),
                    (None, None))
        else:
            if p in m.P1_off:
                diam_selected = next((d for d in m.D if pyo.value(m.b_diam_off_P1[p, d]) > 0.5), None)
            else:
                diam_selected = next((d for d in m.D if pyo.value(m.b_diam_off_P2[p, d, w]) > 0.5), None)
        if diam_selected is None:
            continue

        geom = DATA["pipe_geom"][p]
        L = DATA["L"][p]
        Lh = DATA["Lh"][p] if is_on else 0.0

        geom_draw = _offset_linestring(geom, _LANE_OFFSET_DEG) if len(active_by_route[_route_key(p)]) > 1 else geom

        insulated = is_on and u_selected is not None and abs(u_selected - 0.43) < 1e-9
        if insulated:
            ax.plot(*geom_draw.xy, color=INSULATION_COLOR, linewidth=7, zorder=3.9, solid_capstyle="round")
        ax.plot(*geom_draw.xy, color=_diameter_color(diam_selected), linewidth=3, zorder=4, solid_capstyle="round")

        # Diameter label at the pipe's midpoint, nudged to a nearby free spot.
        label = f'⌀ {diam_selected}'
        mid_pt = geom_draw.interpolate(0.5, normalized=True)
        offsets = [(15, 0), (0, -12), (-15, 0), (0, 12), (15, 12), (15, -12), (-15, 12), (-15, -12)]

        def overlaps(pt, tol=0.15):
            if any(pt.distance(npt) < tol for npt in node_points):
                return True
            if pt.distance(geom_draw) < tol:
                return True
            return any(pt.distance(c) < tol for c in label_centers)

        chosen_offset = offsets[-1]
        for dx_pt, dy_pt in offsets:
            cand = ax.transData.inverted().transform(
                ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array([dx_pt, dy_pt]))
            if not overlaps(Point(cand)):
                chosen_offset = (dx_pt, dy_pt)
                break
        label_centers.append(Point(ax.transData.inverted().transform(
            ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array(chosen_offset))))

        ax.annotate(label, xy=(mid_pt.x, mid_pt.y), xytext=chosen_offset, textcoords='offset points',
                    ha='left', va='bottom' if chosen_offset[1] > 0 else 'top',
                    fontsize=8, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75, ec='none'),
                    arrowprops=dict(arrowstyle='-', color='white', linewidth=0.6, alpha=0.8),
                    zorder=8)

        # Booster station: placed at the pipe's simulated highest/critical
        # point (fraction Lh/L along its length; 0 for offshore, matching
        # developed_plots.py's Lh=0.0 for offshore), no pressure/temperature
        # profile needed to locate it.
        if is_on:
            n_boost = ((pyo.value(m.brep_on1_P1[p, t_match]) + pyo.value(m.brep_on2_P1[p, t_match]))
                       if p in m.P1_on else
                       (pyo.value(m.brep_on1_P2[p, t_match, w]) + pyo.value(m.brep_on2_P2[p, t_match, w])))
        else:
            n_boost = (pyo.value(m.brep_off_P1[p, t_match]) if p in m.P1_off
                       else pyo.value(m.brep_off_P2[p, t_match, w]))

        if n_boost > 0.5:
            frac_high = 0.0 if L <= 0 else float(np.clip(Lh / max(L, 1e-9), 0.0, 1.0))
            eps = 0.001
            p0 = geom_draw.interpolate(max(0.0, frac_high - eps), normalized=True)
            p1 = geom_draw.interpolate(min(1.0, frac_high + eps), normalized=True)
            booster_xy = (p0.x, p0.y)
            angle = np.arctan2(p1.y - p0.y, p1.x - p0.x)
            patch = booster_patch()
            cosphi = np.cos(np.deg2rad(booster_xy[1]))
            scale_deg = 0.08
            trans = (mtrans.Affine2D().rotate(angle).scale(scale_deg / cosphi, scale_deg)
                     .translate(*booster_xy) + ax.transData)
            patch.set_transform(trans)
            patch.set_zorder(7)
            patch.set_clip_on(False)
            ax.add_patch(patch)


def plot_scenario(checkpoint_path: Path, out_dir: Path) -> bool:
    scenario_key = checkpoint_path.name.replace("_model_checkpoint.dill", "")

    with open(checkpoint_path, "rb") as f:
        obj = dill.load(f)
    m, DATA = obj["m"], obj["DATA"]

    # The two legacy benchmark runs (LPBM/SCBM) predate the insulation
    # feature: no U (insulation U-value) dimension on b_diam_on_P2, so
    # add_network()'s diameter lookup doesn't apply. Same exclusion as
    # booster_cause_analysis.py, for the same reason -- pre-existing maps
    # for these two already exist in analysis/results_plot/.
    if not hasattr(m, "U"):
        print(f"Skipping {scenario_key} -- legacy pre-insulation-feature schema")
        return False

    print(f"Plotting {scenario_key} ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_basemap(ax)
    draw_candidate_topology(ax, DATA)
    add_nodes(ax, DATA)
    add_network(ax, DATA, m, year=YEAR, w=SCENARIO)

    # add_nodes() already built a legend (node types, booster icon, insulated
    # pipeline) from proxy handles that were never added as axes children,
    # so get_legend_handles_labels() can't see them -- pull the handles back
    # off the Legend artist itself instead, and extend with the candidate
    # pipeline key (built pipelines are now diameter-colored, explained by
    # the colorbar below rather than a single legend swatch). add_nodes()
    # hardcodes its "Insulated onshore pipeline" swatch to gold, so recolor
    # that one handle to match INSULATION_COLOR (red) instead of touching
    # developed_plots.py itself.
    handles = list(ax.get_legend().legend_handles)
    for h in handles:
        if h.get_label() == "Insulated onshore pipeline":
            h.set_color(INSULATION_COLOR)
    handles.append(
        mlines.Line2D([], [], color=CANDIDATE_COLOR, linewidth=1.2, alpha=0.6, label="Candidate pipeline (not built)")
    )
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

    sm_d = plt.cm.ScalarMappable(cmap=_diameter_cmap, norm=_diameter_norm)
    sm_d.set_array([])
    cbar_d_ax = fig.add_axes([0.32, 0.075, 0.36, 0.015])
    cbar_d = fig.colorbar(sm_d, cax=cbar_d_ax, orientation="horizontal")
    cbar_d.set_ticks([6, 10, 14, 18, 22, 26, 30, 34, 38, 42])
    cbar_d.set_label("Diameter [inch]", fontsize=8)
    cbar_d.ax.tick_params(labelsize=7)

    # Scenario name (top-left) in place of developed_plots.py's small year
    # badge -- year is fixed at 2050 for every plot here, so it's stated
    # once in the figure title instead of a per-plot badge.
    ax.text(0.02, 0.98, scenario_key, transform=ax.transAxes, ha="left", va="top",
            fontsize=20, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75), zorder=10)
    fig.suptitle(f"Final network ({YEAR}) -- scenario {scenario_key}", fontsize=11, y=0.97)

    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / f"{scenario_key}_final_network_{YEAR}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outpath}")
    return True


def main():
    files = sorted(CHECKPOINT_DIR.glob("*_model_checkpoint.dill"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    skipped = [f.name.replace("_model_checkpoint.dill", "") for f in files if not plot_scenario(f, OUT_DIR)]
    if skipped:
        print(f"\nSkipped (legacy schema, see analysis/results_plot/ for their earlier maps): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
