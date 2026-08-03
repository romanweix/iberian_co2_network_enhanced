# developed_plots.py

from pathlib import Path as PPath
import matplotlib as mpl
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import pandas as pd, geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
import matplotlib.transforms as mtrans
import pyomo.environ as pyo
from pyomo.environ import value

# ---------------------------------------------------------------------------
# Base folder for outputs
# ---------------------------------------------------------------------------
_OUTDIR = PPath("output_developed_plots")
_OUTDIR.mkdir(exist_ok=True)

# Mapping from scenario name to a short suffix
_SCEN_SUFFIX = {
    "low_utilization":  "LUS",
    "base_utilization": "EUS",
    "high_utilization": "HUS",
}
def _suffix_for(w: str) -> str:
    return _SCEN_SUFFIX.get(str(w), str(w)[:2].upper())

def _theta_str(m, theta_override: float | None = None) -> str:
    """
    Tries to read the THETA value from the model (m.THETA or m.theta, as a Param or float)
    to include it in the subfolder name. If it doesn't exist,
    uses theta_override or 0 as a default value.
    """
    if theta_override is not None:
        return f"{float(theta_override):.2f}"
    # Preferred Param
    if hasattr(m, "THETA"):
        try:
            return f"{float(pyo.value(m.THETA)):.2f}"
        except Exception:
            pass
    if hasattr(m, "theta"):
        try:
            return f"{float(pyo.value(m.theta)):.2f}"
        except Exception:
            try:
                return f"{float(getattr(m, 'theta')):.2f}"
            except Exception:
                pass
    return "0"

# ---------------------------------------------------------------------------
# BASE MAP OF RELIEF --------------------------------------------------------
# ---------------------------------------------------------------------------
def draw_relief(ax, elev, extent, alpha_land=0.7):
    """
    Draws the relief (no hillshading) on the axis *ax*.
    • elev   -> 2D array of elevations (already rescaled)
    • extent -> (xmin, xmax, ymin, ymax)
    """
    relief_colors = [
        (0.00, '#15511d'),
        (0.05, '#4fa83d'),
        (0.15, '#c7c659'),
        (0.25, '#d9a66b'),
        (0.50, '#a1733b'),
        (0.75, "#ffffff"),
        (1.00, "#aef0ff"),
    ]
    cmap_relief = plt.cm.colors.LinearSegmentedColormap.from_list(
        'relief', relief_colors, N=256
    )
    SEA_BLUE = np.array(plt.cm.colors.to_rgb("#bcd8ff"))

    norm        = plt.cm.colors.Normalize(vmin=0, vmax=np.nanmax(elev))
    rgb_colors  = cmap_relief(norm(elev))[:, :, :3]

    sea_mask = (elev <= 0) | np.isnan(elev)
    rgb_colors[sea_mask] = SEA_BLUE

    alpha = np.full(elev.shape, alpha_land, dtype=float)   # land
    alpha[sea_mask] = 1.0                                  # opaque sea

    rgba = np.dstack((rgb_colors, alpha))                  # final RGBA

    ax.imshow(rgba, extent=extent, origin='upper', zorder=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

# ---------------------------------------------------------------------------
# OVERLAYS: NODES AND PIPES -------------------------------------------------
# ---------------------------------------------------------------------------
def add_nodes(ax, DATA):
    E, S, K, A, M = DATA["E"], DATA["S"], DATA["K"], DATA["A"], DATA["M"]
    stage_n       = DATA["stage_n"]        # {node_id: "First"/"Second"/...}

    # coords -> GeoDataFrame
    df = pd.DataFrame.from_dict(DATA["coords"], orient="index", columns=["lon", "lat"])
    df["node_id"]  = df.index.astype(int)
    df["geometry"] = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def classify(i):
        if i in E:
            return "source"
        if i in S:
            return "sink_uncertain" if stage_n.get(i) == "Second" else "sink"
        if i in K:
            return "util_uncertain" if stage_n.get(i) == "Second" else "util"
        if i in A:
            return "aux"
        if i in M:
            return "sink_uncertain"   # treat trading nodes as uncertain sinks
        return "cities"

    gdf["kind"] = gdf["node_id"].apply(classify)

    style = {
        "source":          dict(color='#A30000', size=40, label="Source nodes"),
        "sink":            dict(color="#228B22", size=40, label="Sink nodes"),
        "sink_uncertain":  dict(color="#B117FF", size=40, label="Uncertain sink nodes"),
        "util_uncertain":  dict(color="#5A3A25", size=40, label="Uncertain utilization nodes"),
        "aux":             dict(color="#D4A017", size=20, label="Auxiliary nodes"),
    }

    handles = []
    for k, sub in gdf.groupby("kind"):
        st = style[k]
        sub.plot(ax=ax,
                 marker="o",
                 facecolor="none",
                 edgecolor=st["color"],
                 markersize=st["size"],
                 zorder=5)
        handles.append(
            mlines.Line2D([], [], ls="",
                          marker="o",
                          markerfacecolor="none",
                          markeredgecolor=st["color"],
                          markersize=8,
                          label=st["label"])
        )

    handles.append(booster_legend_handle())
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

def add_pipes(ax, DATA, model, year=2050, w: str="base_utilization"):
    """
    1) Draws all candidate pipelines in gray.
    2) Overlays the ACTIVE pipelines from scenario w and year 'year'.
    """
    m = model

    # 1) gray background
    gdf = gpd.GeoDataFrame(
        {
            "pipe_id": list(DATA["pipe_geom"].keys()),
            "stage":   [DATA["stage_p"][p] for p in DATA["pipe_geom"]],
            "geometry": list(DATA["pipe_geom"].values()),
        },
        geometry="geometry", crs="EPSG:4326"
    )
    for _, row in gdf.iterrows():
        ls = "-" if row.stage == "First" else (0, (6, 10))
        x, y = row.geometry.xy
        ax.plot(x, y, color="gray", linewidth=0.6, alpha=0.4,
                linestyle=ls, zorder=2)

    # 2) overlay using results
    t_match = next(t for t in m.T if int(t) == year)

    for p in list(m.P_on) + list(m.P_off):
        is_on = (p in m.P_on)

        # A pipeline is active if it has been built and carries flow at some time step
        if is_on:
            active = pyo.value(m.act_on[p, t_match, w]) > 0.5

            # Check that the pipeline carries positive flow in at least one time step
            has_flow_any_t = any(pyo.value(m.q_on[p, tt, w]) > 0.1 for tt in m.T)
        else:
            active = (pyo.value(m.act_off[p, t_match, w]) +
                    (pyo.value(m.b_ship[p, t_match, w]) if p in m.P_off else 0.0)) > 0.5

            # Check that the (offshore) pipeline carries positive flow in at least one time step
            has_flow_any_t = any(pyo.value(m.q_off[p, tt, w]) > 0.1 for tt in m.T)

        if (not active) or (not has_flow_any_t):
            continue

        # Endpoint pressures
        pi_init  = pyo.value(m.pi_orig[p, t_match, w])
        pi_final = pyo.value(m.pi_dest[p, t_match, w])

        # Selected diameter (P1 without w; P2 with w)
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

        # Pressure drops (use DATA tables in both cases)
        dp_fric_high = DATA["dP_frict_high"][(p, diam_selected)]
        dp_fric_far  = DATA["dP_frict_far"] [(p, diam_selected)]
        dp_elev_high = DATA["dP_elev_high"][p]
        dp_elev_far  = DATA["dP_elev_far"] [p]

        p_high  = pi_init - (dp_fric_high + dp_elev_high)

        geom = DATA["pipe_geom"][p]
        L    = DATA["L"][p]
        Lh   = DATA["Lh"][p] if is_on else 0.0

        segs, cols, booster_xy, booster_dir = _pipe_segments_with_gradient(
            geom, pi_init, p_high, pi_final, L, Lh
        )
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=3, zorder=4))

        # ------------------------------------------------------------------
        # Diameter label [inch]
        # ------------------------------------------------------------------
        # diam is already defined (b_diam_on/off)
        label = f'⌀ {diam_selected}'                          # e.g., 30"

        mid_pt = geom.interpolate(0.5, normalized=True)       # midpoint
        label_centers = []

        # --- list of offsets to try (in points) ----------------------------
        offsets = [(15, 0),
                (0, -12),
                (-15, 0),
                (0, 12),
                (15, 12),
                (15, -12),
                (-15, 12),
                (-15, -12)]

        # --- prepare geometries to check -----------------------------------
        # nodes in a quick GeoSeries (coordinates already in DATA['coords']):
        node_points = [Point(xy) for xy in DATA["coords"].values()]

        # active pipelines (only the current p) as a list of LineString
        active_lines = []               # filled once per pipeline
        if active and has_flow_any_t:
            active_lines.append(geom)   # geom is the LineString of this pipeline

        def overlaps(pt, tol=0.15):
            """Return True if 'pt' is too close to nodes or pipelines."""
            # distance to nodes
            for npt in node_points:
                if pt.distance(npt) < tol:
                    return True
            # distance to lines
            for line in active_lines:
                if pt.distance(line) < tol:
                    return True
            for c in label_centers:
                if pt.distance(c) < tol:
                    return True
            return False


        # --- choose a free offset ------------------------------------------
        for dx_pt, dy_pt in offsets:
            # candidate coordinate: shift the midpoint in points and transform to data coords
            cand = ax.transData.inverted().transform(
                    ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array([dx_pt, dy_pt]))
            cand_pt = Point(cand)
            if not overlaps(cand_pt):
                chosen_offset = (dx_pt, dy_pt)
                break
        else:
            chosen_offset = offsets[-1]          # if all collide, use the last one

        label_centers.append(Point(         # store the accepted position
            ax.transData.inverted().transform(
                ax.transData.transform((mid_pt.x, mid_pt.y))
                + np.array(chosen_offset)
            )
        ))

        # --- note -----------------------------------------------------------
        ax.annotate(label,
                    xy=(mid_pt.x, mid_pt.y),
                    xytext=chosen_offset, textcoords='offset points',
                    ha='left', va='bottom' if chosen_offset[1] > 0 else 'top',
                    fontsize=8, color='black',
                    bbox=dict(boxstyle='round,pad=0.15',
                            fc='white', alpha=0.75, ec='none'),
                    arrowprops=dict(arrowstyle='-',
                            color='white',
                            linewidth=0.6,
                            alpha=0.8),
                    zorder=8)

        # Booster: if a booster is active at (t,w) and we have an estimated point
        if is_on:
            n_boost = ( (pyo.value(m.brep_on1_P1[p, t_match]) + pyo.value(m.brep_on2_P1[p, t_match])) if p in m.P1_on
                        else (pyo.value(m.brep_on1_P2[p, t_match, w]) + pyo.value(m.brep_on2_P2[p, t_match, w])) )
        else:
            n_boost = ( pyo.value(m.brep_off_P1[p, t_match]) if p in m.P1_off
                        else pyo.value(m.brep_off_P2[p, t_match, w]) )

        if booster_xy is not None and n_boost > 0.5:
            dx, dy = booster_dir
            angle = np.arctan2(dy, dx)
            patch = booster_patch()
            lat0 = booster_xy[1]
            cosphi = np.cos(np.deg2rad(lat0))
            scale_deg = 0.08
            x_scale, y_scale = (scale_deg / cosphi), scale_deg
            trans = (mtrans.Affine2D()
                     .rotate(angle)
                     .scale(x_scale, y_scale)
                     .translate(*booster_xy)
                     + ax.transData)
            patch.set_transform(trans)
            patch.set_zorder(7)
            patch.set_clip_on(False)
            ax.add_patch(patch)

# -----------------------------------------------------------------------
# GLOBAL pressure colormap: 100 -> 220 bar (blue-red)
# -----------------------------------------------------------------------
pressure_colors = [
    (0.0000, "#0000A3"),   # 100 bar
    (0.1042, "#5BBFFD"),   # 112.5 bar
    (0.2083, "#F57ACA"),   # 125 bar
    (0.3125, "#F85986"),   # 137.5 bar
    (0.4167, "#A30000"),   # 150 bar
    (1.0000, "#270000"),   # 220 bar
]
_pressure_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "pressure_grad", pressure_colors, N=256
)
_pressure_norm = mpl.colors.Normalize(vmin=100, vmax=220)

# -----------------------------------------------------------------------
# Pressure COLORBAR: 100 -> 160 bar (blue-red)
# -----------------------------------------------------------------------
pressure_colorbar = [
    (0.000, "#0000A3"),
    (0.225, "#5BBFFD"),
    (0.450, "#F57ACA"),
    (0.675, "#F85986"),
    (0.900, "#A30000"),
    (1.000, "#270000"),
]
_pressure_cmap_colorbar = mpl.colors.LinearSegmentedColormap.from_list(
    "pressure_grad", pressure_colorbar, N=256
)

def _pressure_color(p_bar):
    """Return the RGBA color for a given pressure [bar]."""
    return _pressure_cmap(_pressure_norm(p_bar))

def _pipe_segments_with_gradient(geom, p_init, p_high, p_final, L, Lh,
                                 boost_delta=50, thresh=100):
    """
    • geom   -> pipeline LineString
    • p_init / p_high / p_final -> pressures (bar)
    • booster_xy = (x, y) of the first segment where pressure drops <= thresh, or None
    • booster_dir indicates the booster orientation, or None
    • Lh     -> distance [km] from start to the highest point
    Returns a list of segments and a list of colors per segment.
    """
    N = 200                                   # ~100 per segment
    total_len_deg = geom.length               # length in degrees (for sampling)
    frac_high = 0.0 if L <= 0 else np.clip(Lh / max(L, 1e-9), 0.0, 1.0)

    # Equally spaced points
    dists = np.linspace(0, total_len_deg, N + 1)
    points = [geom.interpolate(d) for d in dists]

    # Pressure along the line (two-piece profile)
    p_raw = []
    for f in (dists / total_len_deg if total_len_deg > 0 else np.linspace(0, 1, N+1)):
        if f <= frac_high:
            p = p_init + (p_high - p_init) * (0 if frac_high == 0 else (f / frac_high))
        else:
            denom = (1 - frac_high) if (1 - frac_high) > 0 else 1.0
            p = p_high + (p_final - p_high) * ((f - frac_high) / denom)
        p_raw.append(p)

    # Booster: +50 bar from the point where it drops below thresh
    booster_xy = None
    booster_dir = None
    p_vals = p_raw.copy()
    for i in range(len(p_raw) - 1):
        if booster_xy is None and p_raw[i] >= thresh > p_raw[i+1]:
            booster_xy = ((points[i].x + points[i+1].x)/2,
                          (points[i].y + points[i+1].y)/2)
            booster_dir = (points[i+1].x - points[i].x,
                           points[i+1].y - points[i].y)
            p_vals[i+1:] = [p + boost_delta for p in p_vals[i+1:]]
            break

    segs, cols = [], []
    for i in range(N):
        x0, y0 = points[i].x,   points[i].y
        x1, y1 = points[i+1].x, points[i+1].y
        segs.append([[x0, y0], [x1, y1]])
        cols.append(_pressure_color((p_vals[i] + p_vals[i+1]) / 2))

    return segs, cols, booster_xy, booster_dir

def booster_patch(size=40):
    """
    Returns a PathPatch shaped as:
      • a circle of radius 1
      • an inscribed triangle (pointing to the right)
    The patch is centered at (0,0); it will be scaled and rotated later.
    """
    theta = np.linspace(0, 2*np.pi, 33)
    verts_circ = np.column_stack([np.cos(theta), np.sin(theta)])
    codes_circ = [MplPath.MOVETO] + [MplPath.LINETO]*31 + [MplPath.CLOSEPOLY]

    verts_tri = np.array([[ -0.5,  0.8660],
                          [ -0.5, -0.8660],
                          [  1.0,  0.0000],
                          [ -0.5,  0.8660]])
    codes_tri = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]

    verts = np.concatenate([verts_circ, verts_tri])
    codes = codes_circ + codes_tri
    path  = MplPath(verts, codes)

    patch = PathPatch(path,
                      facecolor="none",
                      edgecolor="black",
                      linewidth=0.8)
    patch.set_transform(mtrans.Affine2D().scale(size/2))
    return patch

def booster_marker():
    """Return a marker based on the circle-triangle Path."""
    return booster_patch().get_path()

def booster_legend_handle(markersize=8):
    """Return a Line2D handle ready for the legend."""
    return mlines.Line2D([],
                         [],
                         marker=booster_marker(),
                         markersize=markersize,
                         markerfacecolor="none",
                         markeredgecolor="black",
                         linestyle="",
                         label="Boosting station")

# ---------------------------------------------------------------------------
# Main plotting function (single) and batch (all) ---------------------------
# ---------------------------------------------------------------------------
def plot_network(m, DATA, scenario="base_utilization", year=2050,
                 save=True, show=True, theta_override: float | None = None):
    """
    Generates the map for (scenario, year).
    Displays only this one on screen, but you can call save_all_plots()
    to generate all 15 PNGs.
    """
    import rasterio
    dem_path = "merged_srtm.tif"
    with rasterio.open(dem_path) as src:
        scale = 4
        elev = src.read(1, out_shape=(src.height//scale, src.width//scale))
        xmin, ymin, xmax, ymax = src.bounds
        extent = (xmin, xmax, ymin, ymax)

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_relief(ax, elev, extent)

    # Soft vertical cropping
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin + 0.5, ymax - 1)

    # Nodes and pipelines (scenario/year)
    add_nodes(ax, DATA)
    add_pipes(ax, DATA, m, year=year, w=scenario)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=_pressure_cmap_colorbar,
        norm=mpl.colors.Normalize(vmin=100, vmax=160)
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.075, 0.5, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([100, 110, 120, 130, 140, 150, 160])
    cbar.set_ticklabels(["100", "110", "120", "130", "140", "150", "+150"])
    cbar.set_label("Pressure [bar]")


    # --- Badge showing the time step (top-right) ---------------------------
    # Match the font size to the colorbar label font size:
    try:
        fs = cbar.ax.xaxis.label.get_size()  # horizontal
    except Exception:
        fs = cbar.ax.yaxis.label.get_size()  # in case it is vertical

    ax.text(
        0.98, 0.98,              # top-right corner in axis coordinates
        f"{year}",               # text: the time step (2030, 2035, ...)
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=1.5*fs,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",          # white background
            ec="none",
            alpha=0.7            # ~30% transparency (70% opacity)
        ),
        zorder=10
    )


    if save:
        ths = _theta_str(m, theta_override)
        subdir = _OUTDIR / f"output_developed_plots_theta_{ths}"
        subdir.mkdir(parents=True, exist_ok=True)
        suf = _suffix_for(scenario)
        outpath = subdir / f"developed_solution_{suf}_theta_{ths}_{int(year)}.png"
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"✔  Map saved in {outpath.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def save_all_plots(m, DATA, theta_override: float | None = None):
    """Generates and saves 3×5 plots (without showing them)."""
    years = sorted(int(t) for t in m.T)
    scenarios = list(m.W)  # Pyomo Set
    for w in scenarios:
        for yr in years:
            plot_network(m, DATA, scenario=w, year=yr, save=True, show=False,
                         theta_override=theta_override)
