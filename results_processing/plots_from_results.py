# plots_from_excel.py
"""
Recreates the CO2 transport network maps exclusively from:
    - the Excel results file
    - the simplified_data.py module (for topology and geometry)

"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Point
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
import matplotlib.lines as mlines
import matplotlib.transforms as mtrans
from matplotlib.path import Path as MplPath
import rasterio

# Import structural data (nodes, pipes, geometries, etc.)
from simplified_data import DATA

# ---------------------------------------------------------------------------
# Global folders and scenario configuration
# ---------------------------------------------------------------------------
_OUTDIR = Path("results_summaries")
_OUTDIR.mkdir(exist_ok=True)
_SCENARIOS = {"LUS": "low_utilization", "EUS": "base_utilization", "HUS": "high_utilization"}

# ---------------------------------------------------------------------------
# Relief and visual utilities
# ---------------------------------------------------------------------------
def draw_relief(ax, elev, extent, alpha_land=0.7):
    """Draws shaded land-sea relief as map background."""
    relief_colors = [
        (0.00, '#15511d'), (0.05, '#4fa83d'), (0.15, '#c7c659'),
        (0.25, '#d9a66b'), (0.50, '#a1733b'), (0.75, "#ffffff"), (1.00, "#aef0ff"),
    ]
    cmap_relief = plt.cm.colors.LinearSegmentedColormap.from_list('relief', relief_colors, N=256)
    SEA_BLUE = np.array(plt.cm.colors.to_rgb("#bcd8ff"))
    norm = plt.cm.colors.Normalize(vmin=0, vmax=np.nanmax(elev))
    rgb_colors = cmap_relief(norm(elev))[:, :, :3]
    sea_mask = (elev <= 0) | np.isnan(elev)
    rgb_colors[sea_mask] = SEA_BLUE
    alpha = np.full(elev.shape, alpha_land, dtype=float)
    alpha[sea_mask] = 1.0
    rgba = np.dstack((rgb_colors, alpha))
    ax.imshow(rgba, extent=extent, origin='upper', zorder=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

def booster_patch(size=40):
    """Returns a circular + triangle patch representing a booster station."""
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
    path = MplPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor="black", linewidth=0.8)
    patch.set_transform(mtrans.Affine2D().scale(size/2))
    return patch

def booster_legend_handle(markersize=8):
    """Returns a Line2D object for legend entry of boosters."""
    return mlines.Line2D([],
                         [],
                         marker=booster_patch().get_path(),
                         markersize=markersize,
                         markerfacecolor="none",
                         markeredgecolor="black",
                         linestyle="",
                         label="Boosting station")

# ---------------------------------------------------------------------------
# Pressure color map
# ---------------------------------------------------------------------------
pressure_colors = [
    (0.0000, "#0000A3"), (0.1042, "#5BBFFD"),
    (0.2083, "#F57ACA"), (0.3125, "#F85986"),
    (0.4167, "#A30000"), (1.0000, "#270000"),
]
_pressure_cmap = mpl.colors.LinearSegmentedColormap.from_list("pressure_grad", pressure_colors, N=256)
_pressure_norm = mpl.colors.Normalize(vmin=100, vmax=220)

def _pressure_color(p_bar):
    """Returns RGBA color corresponding to a given pressure value [bar]."""
    return _pressure_cmap(_pressure_norm(p_bar))

# ---------------------------------------------------------------------------
# Core: Add nodes and pipes using Excel data
# ---------------------------------------------------------------------------
def add_nodes(ax, DATA):
    """Draws all network nodes using simplified_data.py."""
    E, S, K, A, M = DATA["E"], DATA["S"], DATA["K"], DATA["A"], DATA["M"]
    stage_n = DATA["stage_n"]
    df = pd.DataFrame.from_dict(DATA["coords"], orient="index", columns=["lon", "lat"])
    df["node_id"] = df.index.astype(int)
    df["geometry"] = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def classify(i):
        if i in E: return "source"
        if i in S: return "sink_uncertain" if stage_n.get(i) == "Second" else "sink"
        if i in K: return "util_uncertain" if stage_n.get(i) == "Second" else "util"
        if i in A: return "aux"
        if i in M: return "sink_uncertain"
        return "other"

    gdf["kind"] = gdf["node_id"].apply(classify)

    style = {
        "source": dict(color='#A30000', size=40, label="Source nodes"),
        "sink": dict(color="#228B22", size=40, label="Sink nodes"),
        "sink_uncertain": dict(color="#B117FF", size=40, label="Uncertain sink nodes"),
        "util_uncertain": dict(color="#5A3A25", size=40, label="Uncertain utilization nodes"),
        "aux": dict(color="#D4A017", size=20, label="Auxiliary nodes"),
    }

    handles = []
    for k, sub in gdf.groupby("kind"):
        if k not in style: continue
        st = style[k]
        sub.plot(ax=ax, marker="o", facecolor="none",
                 edgecolor=st["color"], markersize=st["size"], zorder=5)
        handles.append(mlines.Line2D([], [], ls="", marker="o",
                                     markerfacecolor="none",
                                     markeredgecolor=st["color"],
                                     markersize=8, label=st["label"]))
    handles.append(booster_legend_handle())
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

def _pipe_segments_with_gradient(geom, p_init, p_high, p_final, L, Lh,
                                 boost_delta=50, thresh=100):
    """
    Computes pressure gradient and booster placement along the pipeline.
    Returns:
        segs: list of [[x0,y0],[x1,y1]] line segments
        cols: list of RGBA colors for each segment
        booster_xy: (x, y) coordinate of booster location, or None
        booster_dir: (dx, dy) orientation vector at booster, or None
    """
    N = 200
    total_len_deg = geom.length
    frac_high = 0.0 if L <= 0 else np.clip(Lh / max(L, 1e-9), 0.0, 1.0)

    dists = np.linspace(0, total_len_deg, N + 1)
    points = [geom.interpolate(d) for d in dists]

    # Pressure profile (two-stage: rise to p_high, then drop to p_final)
    p_raw = []
    for f in (dists / total_len_deg if total_len_deg > 0 else np.linspace(0, 1, N+1)):
        if f <= frac_high:
            p = p_init + (p_high - p_init) * (0 if frac_high == 0 else (f / frac_high))
        else:
            denom = (1 - frac_high) if (1 - frac_high) > 0 else 1.0
            p = p_high + (p_final - p_high) * ((f - frac_high) / denom)
        p_raw.append(p)

    # Booster placement (add +boost_delta bar once below threshold)
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

    # Build colored segments
    segs, cols = [], []
    for i in range(N):
        x0, y0 = points[i].x,   points[i].y
        x1, y1 = points[i+1].x, points[i+1].y
        segs.append([[x0, y0], [x1, y1]])
        cols.append(_pressure_color((p_vals[i] + p_vals[i+1]) / 2))

    return segs, cols, booster_xy, booster_dir


def add_pipes_from_excel(ax, df_pipes, DATA, year):

    # --- 1) Draw all candidate pipes in gray (background) ------------------
    gdf = gpd.GeoDataFrame({
        "pipe_id": list(DATA["pipe_geom"].keys()),
        "stage": [DATA["stage_p"][p] for p in DATA["pipe_geom"]],
        "geometry": list(DATA["pipe_geom"].values()),
    }, geometry="geometry", crs="EPSG:4326")

    for _, row in gdf.iterrows():
        ls = "-" if row.stage == "First" else (0, (6, 10))
        x, y = row.geometry.xy
        ax.plot(x, y, color="gray", linewidth=0.6, alpha=0.4,
                linestyle=ls, zorder=2)

    # --- 2) Define time steps and flow columns -----------------------------
    flow_cols = {
        2030: "Flow in 2030",
        2035: "Flow in 2035",
        2040: "Flow in 2040",
        2045: "Flow in 2045",
        2050: "Flow in 2050",
    }
    valid_years = [y for y in flow_cols if y <= year]

    # --- 3) Draw active pipes ----------------------------------------------
    for _, row in df_pipes.iterrows():
        if not bool(row.get("Installed", 0)):
            continue

        # Check installation condition (installed and already built by this year)
        installation_year = row.get("Installation Year", None)
        if installation_year is None or pd.isna(installation_year):
            continue
        if installation_year > year:
            continue

        pid = str(row["Pipe ID"])
        geom = DATA["pipe_geom"].get(pid)
        if geom is None:
            continue

        # --- 4) Read pressures and parameters ------------------------------
        p_init = row.get("Initial pressure [bar]", np.nan)
        p_high = row.get("Pressure at highest point [bar]", np.nan)
        p_final = row.get("Final pressure [bar]", np.nan)
        L = row.get("Longitude [km]", np.nan)
        Lh = row.get("Distance until highest point [km]", np.nan)
        n_boost = row.get("Number of boosters", 0)

        # --- Fix for offshore pipes: if missing elevation info, treat as flat ---
        if pd.isna(Lh) or Lh <= 0:
            Lh = 0.0
            p_high = p_init  # no elevation gain offshore

        # --- 5) Compute pressure gradient + booster location ---------------
        segs, cols, booster_xy, booster_dir = _pipe_segments_with_gradient(
            geom, p_init, p_high, p_final, L, Lh
        )
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=3, zorder=4))

        # --- 6) Add diameter label (⌀ XX") with automatic offset ------------
        diam_val = row.get("Diameter [inch]", "")
        try:
            diam_val = int(round(float(diam_val)))
        except Exception:
            pass
        label = f'⌀ {diam_val}'
        mid_pt = geom.interpolate(0.5, normalized=True)
        label_centers = []

        # Offsets to test (in points)
        offsets = [(15, 0), (0, -12), (-15, 0), (0, 12),
                   (15, 12), (15, -12), (-15, 12), (-15, -12)]

        # Precompute nearby geometries for overlap avoidance
        node_points = [Point(xy) for xy in DATA["coords"].values()]
        active_lines = [geom]

        def overlaps(pt, tol=0.15):
            """Check if label overlaps with nodes, lines or other labels."""
            for npt in node_points:
                if pt.distance(npt) < tol:
                    return True
            for line in active_lines:
                if pt.distance(line) < tol:
                    return True
            for c in label_centers:
                if pt.distance(c) < tol:
                    return True
            return False

        # Choose the first free offset
        for dx_pt, dy_pt in offsets:
            cand = ax.transData.inverted().transform(
                ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array([dx_pt, dy_pt])
            )
            cand_pt = Point(cand)
            if not overlaps(cand_pt):
                chosen_offset = (dx_pt, dy_pt)
                break
        else:
            chosen_offset = offsets[-1]

        label_centers.append(Point(
            ax.transData.inverted().transform(
                ax.transData.transform((mid_pt.x, mid_pt.y))
                + np.array(chosen_offset)
            )
        ))

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

        # --- 7) Draw booster icon with correct orientation -----------------
        if n_boost and n_boost > 0 and booster_xy is not None:
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

# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_from_excel(excel_path, scenario="EUS", year=2050, save=True, show=True):
    """Plots one scenario-year map using Excel results and simplified_data."""
    xls = pd.ExcelFile(excel_path)
    sheet_name = f"{scenario} - Pipes"
    df_pipes = pd.read_excel(xls, sheet_name=sheet_name)

    # Load terrain relief (DEM)
    dem_path = "merged_srtm.tif"
    with rasterio.open(dem_path) as src:
        scale = 4
        elev = src.read(1, out_shape=(src.height // scale, src.width // scale))
        xmin, ymin, xmax, ymax = src.bounds
        extent = (xmin, xmax, ymin, ymax)

    # create base figure
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_relief(ax, elev, extent)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin + 0.5, ymax - 1)

    # add nodes and pipes
    add_nodes(ax, DATA)
    add_pipes_from_excel(ax, df_pipes, DATA, year)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=_pressure_cmap,
                               norm=mpl.colors.Normalize(vmin=100, vmax=160))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.075, 0.5, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([100, 110, 120, 130, 140, 150, 160])
    cbar.set_ticklabels(["100", "110", "120", "130", "140", "150", "+150"])
    cbar.set_label("Pressure [bar]")

    # year badge (top-right)
    fs = cbar.ax.xaxis.label.get_size()
    ax.text(0.98, 0.98, f"{year}", transform=ax.transAxes,
            ha="right", va="top", fontsize=1.5 * fs, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            zorder=10)

    # save or display
    if save:
        # Create output directory for this value of theta (e.g. plots_theta_0.80)
        theta_str = excel_path.stem.split("_theta_")[-1].replace(".xlsx", "")
        plots_dir = Path(f"plots_theta_{theta_str}")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save plot inside the corresponding folder
        outpath = plots_dir / f"plot_{scenario}_{year}.png"
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"✔  Saved: {outpath.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------------------------------------------------------------------------
# Batch generation for all scenarios and years
# ---------------------------------------------------------------------------
def save_all_plots_from_excel(excel_path):
    """Generates all 3×5 plots (LUS, EUS, HUS × years 2030–2050)."""
    years = [2030, 2035, 2040, 2045, 2050]
    for scenario in _SCENARIOS.keys():
        for yr in years:
            plot_from_excel(excel_path, scenario=scenario, year=yr, save=True, show=False)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --------------------------------------------------------------
    # Set the theta value manually here
    # --------------------------------------------------------------
    THETA = 1.00   # <-- Change this value manually each time you run the script

    # Build the Excel filename automatically based on THETA
    excel_filename = f"stochastic_results_theta_{THETA}0.xlsx"
    excel_file = _OUTDIR / excel_filename

    # Check that the file exists
    if not excel_file.exists():
        raise FileNotFoundError(f"❌ Excel file not found: {excel_file}")

    # Generate and save all plots for this value of THETA
    print(f"📊 Generating plots for θ = {THETA}0")
    save_all_plots_from_excel(excel_file)
    print(f"✅ All plots generated successfully for θ = {THETA}0")
