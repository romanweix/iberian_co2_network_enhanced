# theta_comparison_plots.py
"""
Generates a figure comparing two θ (THETA) values for the same scenario and year.
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Point
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
import matplotlib.lines as mlines
import matplotlib.transforms as mtrans
from matplotlib.path import Path as MplPath

# Import topology and geometries
from simplified_data import DATA

# ---------------------------------------------------------------------------
# === USER INPUT ===
# ---------------------------------------------------------------------------
SCENARIO = "EUS"          # Choose among: "LUS", "EUS", "HUS"
YEAR = 2050               # One of [2030, 2035, 2040, 2045, 2050]
THETA_1 = 0.40            # e.g. 0.00, 0.10, 0.20, 0.30 ..., 1.00
THETA_2 = 0.90            # second theta for comparison

# ---------------------------------------------------------------------------
# Global paths
# ---------------------------------------------------------------------------
_OUTDIR = Path("results_summaries")
_SCENARIOS = {"LUS": "low_utilization", "EUS": "base_utilization", "HUS": "high_utilization"}

# ---------------------------------------------------------------------------
# Relief utilities
# ---------------------------------------------------------------------------
def draw_relief(ax, elev, extent, alpha_land=0.7):
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
    return _pressure_cmap(_pressure_norm(p_bar))

# ---------------------------------------------------------------------------
# Booster and legend utilities
# ---------------------------------------------------------------------------
def booster_patch(size=40):
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
    return mlines.Line2D([], [], marker=booster_patch().get_path(),
                         markersize=markersize, markerfacecolor="none",
                         markeredgecolor="black", linestyle="",
                         label="Boosting station")

def create_legend_handles():
    style = {
        "source": dict(color='#A30000', label="Source nodes"),
        "sink": dict(color="#228B22", label="Sink nodes"),
        "sink_uncertain": dict(color="#B117FF", label="Uncertain sink nodes"),
        "util_uncertain": dict(color="#5A3A25", label="Uncertain utilization nodes"),
        "aux": dict(color="#D4A017", label="Auxiliary nodes"),
    }
    handles = []
    for st in style.values():
        handles.append(mlines.Line2D([], [], ls="", marker="o",
                                     markerfacecolor="none",
                                     markeredgecolor=st["color"],
                                     markersize=8, label=st["label"]))
    handles.append(booster_legend_handle())
    return handles

# ---------------------------------------------------------------------------
# Node and pipe drawing
# ---------------------------------------------------------------------------
def add_nodes(ax, DATA):
    E, S, K, A, M = DATA["E"], DATA["S"], DATA["K"], DATA["A"], DATA["M"]
    stage_n = DATA["stage_n"]
    df = pd.DataFrame.from_dict(DATA["coords"], orient="index", columns=["lon", "lat"])
    df["geometry"] = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def classify(i):
        if i in E: return "source"
        if i in S: return "sink_uncertain" if stage_n.get(i) == "Second" else "sink"
        if i in K: return "util_uncertain" if stage_n.get(i) == "Second" else "util"
        if i in A: return "aux"
        if i in M: return "sink_uncertain"
        return "other"

    gdf["kind"] = [classify(i) for i in gdf.index]
    style = {
        "source": dict(color='#A30000', size=30),
        "sink": dict(color="#228B22", size=30),
        "sink_uncertain": dict(color="#B117FF", size=30),
        "util_uncertain": dict(color="#5A3A25", size=30),
        "aux": dict(color="#D4A017", size=20),
    }
    for k, sub in gdf.groupby("kind"):
        if k not in style: continue
        st = style[k]
        sub.plot(ax=ax, marker="o", facecolor="none",
                 edgecolor=st["color"], markersize=st["size"], zorder=5)

# --- Pipes (copied from your version, unmodified except readability) ---
def _pipe_segments_with_gradient(geom, p_init, p_high, p_final, L, Lh,
                                 boost_delta=50, thresh=100):
    N = 200
    total_len_deg = geom.length
    frac_high = 0.0 if L <= 0 else np.clip(Lh / max(L, 1e-9), 0.0, 1.0)
    dists = np.linspace(0, total_len_deg, N + 1)
    points = [geom.interpolate(d) for d in dists]
    p_raw = []
    for f in (dists / total_len_deg if total_len_deg > 0 else np.linspace(0, 1, N+1)):
        if f <= frac_high:
            p = p_init + (p_high - p_init) * (0 if frac_high == 0 else (f / frac_high))
        else:
            denom = (1 - frac_high) if (1 - frac_high) > 0 else 1.0
            p = p_high + (p_final - p_high) * ((f - frac_high) / denom)
        p_raw.append(p)

    booster_xy = None; booster_dir = None
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
        x0, y0 = points[i].x, points[i].y
        x1, y1 = points[i+1].x, points[i+1].y
        segs.append([[x0, y0], [x1, y1]])
        cols.append(_pressure_color((p_vals[i] + p_vals[i+1]) / 2))
    return segs, cols, booster_xy, booster_dir

def add_pipes_from_excel(ax, df_pipes, DATA, year):
    gdf = gpd.GeoDataFrame({
        "pipe_id": list(DATA["pipe_geom"].keys()),
        "stage": [DATA["stage_p"][p] for p in DATA["pipe_geom"]],
        "geometry": list(DATA["pipe_geom"].values()),
    }, geometry="geometry", crs="EPSG:4326")

    for _, row in gdf.iterrows():
        ls = "-" if row.stage == "First" else (0, (6, 10))
        x, y = row.geometry.xy
        ax.plot(x, y, color="gray", linewidth=0.6, alpha=0.4, linestyle=ls, zorder=2)

    for _, row in df_pipes.iterrows():
        if not bool(row.get("Installed", 0)):
            continue
        installation_year = row.get("Installation Year", None)
        if installation_year is None or pd.isna(installation_year) or installation_year > year:
            continue

        pid = str(row["Pipe ID"])
        geom = DATA["pipe_geom"].get(pid)
        if geom is None: continue

        p_init = row.get("Initial pressure [bar]", np.nan)
        p_high = row.get("Pressure at highest point [bar]", np.nan)
        p_final = row.get("Final pressure [bar]", np.nan)
        L = row.get("Longitude [km]", np.nan)
        Lh = row.get("Distance until highest point [km]", np.nan)
        n_boost = row.get("Number of boosters", 0)
        if pd.isna(Lh) or Lh <= 0:
            Lh = 0.0
            p_high = p_init

        segs, cols, booster_xy, booster_dir = _pipe_segments_with_gradient(geom, p_init, p_high, p_final, L, Lh)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2, zorder=4))

        # --- diameter label ---
        from shapely.geometry import Point as ShPoint

        diam_val = row.get("Diameter [inch]", "")
        try:
            diam_val = int(round(float(diam_val)))
        except Exception:
            pass
        label = f'⌀ {diam_val}'
        mid_pt = geom.interpolate(0.5, normalized=True)

        # Generates candidate offsets (pixels)
        offsets = [(15, 0), (0, -12), (-15, 0), (0, 12),
                (15, 12), (15, -12), (-15, 12), (-15, -12)]

        node_points = [ShPoint(xy) for xy in DATA["coords"].values()]

        def overlaps(pt, tol=0.2):
            """Checks if point overlaps with nodes or line geometry."""
            for npt in node_points:
                if pt.distance(npt) < tol:
                    return True
            for line in [geom]:
                if pt.distance(line) < tol:
                    return True
            return False

        # Select the first non-overlapping offset
        for dx_pt, dy_pt in offsets:
            cand = ax.transData.inverted().transform(
                ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array([dx_pt, dy_pt])
            )
            cand_pt = ShPoint(cand)
            if not overlaps(cand_pt):
                chosen_offset = (dx_pt, dy_pt)
                break
        else:
            chosen_offset = (10, 10)  # fallback

        # Draw annotation
        ax.annotate(label,
                    xy=(mid_pt.x, mid_pt.y),
                    xytext=chosen_offset, textcoords='offset points',
                    ha='left', va='bottom' if chosen_offset[1] > 0 else 'top',
                    fontsize=6, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75, ec='none'),
                    arrowprops=dict(arrowstyle='-',
                                    color='white',
                                    linewidth=0.6,
                                    alpha=0.8),
                    zorder=8)

        # Booster symbol
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
            ax.add_patch(patch)

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def compare_two_thetas(scenario, year, theta1, theta2, save=True, show=True):
    with rasterio.open("merged_srtm.tif") as src:
        scale = 4
        elev = src.read(1, out_shape=(src.height // scale, src.width // scale))
        xmin, ymin, xmax, ymax = src.bounds
        extent = (xmin, xmax, ymin, ymax)

    thetas = [theta1, theta2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=False)
    plt.subplots_adjust(wspace=0.03, hspace=0.0)

    for ax, theta in zip(axes, thetas):
        excel_filename = f"stochastic_results_theta_{theta:.1f}0.xlsx"
        excel_path = _OUTDIR / excel_filename
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        df_pipes = pd.read_excel(excel_path, sheet_name=f"{scenario} - Pipes")

        draw_relief(ax, elev, extent)
        add_nodes(ax, DATA)
        add_pipes_from_excel(ax, df_pipes, DATA, year)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin + 0.5, ymax - 1)
        ax.text(0.95, 0.95, f"{year}", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7), zorder=10)
        ax.text(0.95, 0.05, f"θ = {theta:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7), zorder=10)
        ax.set_xticks([]); ax.set_yticks([])

    # ---------------------------------------------------------------------------
    # Colorbar + legend (horizontal, centered)
    # ---------------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=_pressure_cmap_colorbar,
                               norm=mpl.colors.Normalize(vmin=100, vmax=160))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.16, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Pressure [bar]", fontsize=11)
    cbar.set_ticks([100, 110, 120, 130, 140, 150, 160])
    cbar.set_ticklabels(["100", "110", "120", "130", "140", "150", "+150"])

    # Legend (one single row)
    handles = create_legend_handles()
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    legend_ax.legend(handles=handles, loc="center", ncol=len(handles),
                     frameon=True, fontsize=9, edgecolor="black")
    legend_ax.axis("off")

    if save:
        outname = f"theta_comparison_{scenario}_{year}_{theta1:.2f}_vs_{theta2:.2f}.png"
        outpath = Path("theta_comparison_plots")
        outpath.mkdir(exist_ok=True)
        fig.savefig(outpath / outname, dpi=300, bbox_inches="tight")
        print(f"✔ Saved comparison figure: {outpath / outname}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# -----------------------------------------------------------------------
# COLORBAR pressure 100 → 160 bar  (blue → red)
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
    """Returns the RGBA color for a given pressure [bar].""" 
    return _pressure_cmap(_pressure_norm(p_bar))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📊 Comparing {SCENARIO} scenario for {YEAR}: θ={THETA_1} vs θ={THETA_2}")
    compare_two_thetas(SCENARIO, YEAR, THETA_1, THETA_2)
    print("✅ Comparison figure created successfully.")
