# plot_final_network_maps_legacy.py
#
# Same final-network map style as plot_final_network_maps.py, for the two
# legacy benchmark checkpoints (LPBM, SCBM) that script skips. Those two
# predate the insulation feature: no m.U dimension, no theta_min/temperature
# tracking at all, and diameter selection is 2-argument
# (b_diam_on_P1[p,d] / b_diam_on_P2[p,d,w]) instead of the modern
# 3-argument (b_diam_on_P1[p,d,u] / b_diam_on_P2[p,d,u,w]). Everything else
# -- basemap, candidate topology, node markers, booster placement, diameter
# labeling/coloring, legend, colorbar, scenario badge -- is identical, so
# this module imports the shared pieces from plot_final_network_maps.py
# rather than duplicating them, and only reimplements the diameter-lookup
# block (add_network -> add_network_legacy) and the top-level driver
# (plot_scenario -> plot_scenario_legacy, without the hasattr(m, "U") skip
# and without the "Insulated onshore pipeline" legend entry, since
# insulation isn't a modeled concept for these two runs).
#
# Usage (from the repo root):
#   python -m analysis.plot_final_network_maps_legacy

import sys
from collections import defaultdict
from pathlib import Path

import dill
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pyomo.environ as pyo
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iberian_co2_network.developed_plots import add_nodes, booster_patch, _offset_linestring, _LANE_OFFSET_DEG  # noqa: E402
from analysis.plot_final_network_maps import (  # noqa: E402
    CHECKPOINT_DIR, OUT_DIR, YEAR, SCENARIO, CANDIDATE_COLOR,
    _diameter_cmap, _diameter_norm, _diameter_color,
    draw_basemap, draw_candidate_topology,
)

LEGACY_SCENARIOS = ["LPBM", "SCBM"]


def add_network_legacy(ax, DATA, model, year, w):
    """Same as plot_final_network_maps.add_network(), but with a legacy-
    schema-compatible diameter lookup (no U/insulation dimension) --
    insulation never highlights here since these runs have no such concept."""
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

        if is_on:
            if p in m.P1_on:
                diam_selected = next((d for d in m.D if pyo.value(m.b_diam_on_P1[p, d]) > 0.5), None)
            else:
                diam_selected = next((d for d in m.D if pyo.value(m.b_diam_on_P2[p, d, w]) > 0.5), None)
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


def plot_scenario_legacy(checkpoint_path: Path, out_dir: Path) -> bool:
    scenario_key = checkpoint_path.name.replace("_model_checkpoint.dill", "")

    with open(checkpoint_path, "rb") as f:
        obj = dill.load(f)
    m, DATA = obj["m"], obj["DATA"]

    print(f"Plotting {scenario_key} (legacy schema) ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_basemap(ax)
    draw_candidate_topology(ax, DATA)
    add_nodes(ax, DATA)
    add_network_legacy(ax, DATA, m, year=YEAR, w=SCENARIO)

    # Same legend-handle-recovery workaround as plot_final_network_maps.py
    # (add_nodes() builds its legend from proxy handles that were never
    # added as axes children). These runs have no insulation concept, so
    # drop that swatch instead of recoloring it.
    handles = [h for h in ax.get_legend().legend_handles
               if h.get_label() != "Insulated onshore pipeline"]
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
    for scenario in LEGACY_SCENARIOS:
        checkpoint_path = CHECKPOINT_DIR / f"{scenario}_model_checkpoint.dill"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found, skipping: {checkpoint_path}")
            continue
        plot_scenario_legacy(checkpoint_path, OUT_DIR)


if __name__ == "__main__":
    main()
