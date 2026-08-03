# traditional_plots.py

from pathlib import Path as PPath
import math
import matplotlib as mpl
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import pandas as pd, geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib.lines as mlines
from matplotlib.patches import PathPatch
import matplotlib.transforms as mtrans
import pyomo.environ as pyo
from pyomo.environ import value
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Carpeta base para outputs
# ---------------------------------------------------------------------------
_OUTDIR = PPath("output_traditional_plots")
_OUTDIR.mkdir(exist_ok=True)

# Mapeo de nombre de escenario a sufijo corto
_SCEN_SUFFIX = {
    "low_utilization":  "LUS",
    "base_utilization": "EUS",
    "high_utilization": "HUS",
}
def _suffix_for(w: str) -> str:
    return _SCEN_SUFFIX.get(str(w), str(w)[:2].upper())

def _theta_str(m, theta_override: float | None = None) -> str:
    """
    Try to read THETA from the model to embed in folder/file names.
    Falls back to theta_override or 0.
    """
    if theta_override is not None:
        return f"{float(theta_override):.2f}"
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
    return "0.00"

# ---------------------------------------------------------------------------
# BASE MAP OF RELIEF --------------------------------------------------------
# ---------------------------------------------------------------------------
def draw_relief(ax, elev, extent, alpha_land=0.7):
    """
    Draw a simple relief (no hillshade) as a background layer.
    • elev   → 2-D elevation array (already rescaled)
    • extent → (xmin, xmax, ymin, ymax)
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
    alpha[sea_mask] = 1.0                                  # sea opaque

    rgba = np.dstack((rgb_colors, alpha))

    ax.imshow(rgba, extent=extent, origin='upper', zorder=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

# ---------------------------------------------------------------------------
# OVERLAYS: NODES & PIPES ---------------------------------------------------
# ---------------------------------------------------------------------------
def add_nodes(ax, DATA):
    E, S, K, A, M = DATA["E"], DATA["S"], DATA["K"], DATA["A"], DATA["M"]
    stage_n       = DATA["stage_n"]        # {node_id: "First"/"Second"/…}

    # coords → GeoDataFrame
    df = pd.DataFrame.from_dict(DATA["coords"], orient="index", columns=["lon", "lat"])
    df["node_id"]  = df.index.astype(int)
    df["geometry"] = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def classify(i):
        if i in E:               return "source"
        if i in S:
            return "sink_uncertain" if stage_n.get(i) == "Second" else "sink"
        if i in K:
            return "util_uncertain" if stage_n.get(i) == "Second" else "util"
        if i in A:               return "aux"
        if i in M:               return "sink_uncertain"   # trading as uncertain sink
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

    # # --- tags with node ID (below marker) ----------------
    # for _, row in gdf.iterrows():
    #     x, y = row.geometry.x, row.geometry.y
    #     ax.annotate(
    #         str(int(row["node_id"])),
    #         xy=(x, y),
    #         xytext=(0, -9), textcoords="offset points",
    #         ha="center", va="top",
    #         fontsize=7, color="black",
    #         zorder=6,
    #         path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.9)]
    #     )


# --- helpers for traditional boosters placement ---------------------------

def _n_boosters_from_length_km(L_km: float, km_per_boost: float = 150.0) -> int:
    """English: 0 if L ≤ step; 1 if step < L ≤ 2*step; 2 if 2*step < L ≤ 3*step; etc."""
    return max(0, math.ceil(L_km / km_per_boost) - 1)

def _tangent_dir_at(geom, s_norm: float, eps: float = 1e-4) -> tuple[float, float]:
    """English: approximate tangent vector (dx, dy) at normalized curvilinear abscissa s."""
    s0 = max(0.0, min(1.0, s_norm - eps))
    s1 = max(0.0, min(1.0, s_norm + eps))
    p0 = geom.interpolate(s0, normalized=True)
    p1 = geom.interpolate(s1, normalized=True)
    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    if abs(dx) + abs(dy) < 1e-12:
        # fallback: try a bigger epsilon
        s0 = max(0.0, min(1.0, s_norm - 5*eps))
        s1 = max(0.0, min(1.0, s_norm + 5*eps))
        p0 = geom.interpolate(s0, normalized=True)
        p1 = geom.interpolate(s1, normalized=True)
        dx, dy = (p1.x - p0.x), (p1.y - p0.y)
        if abs(dx) + abs(dy) < 1e-12:
            dx, dy = 1.0, 0.0
    return dx, dy

def _traditional_booster_positions(geom, L_km: float, km_per_boost: float = 150.0):
    """
    English:
    Return a list of booster positions along the pipe, starting from the origin,
    at distances {150, 300, 450, ...} km strictly less than the total L_km.
    Each element is (x, y, dx, dy) where (dx, dy) is the local direction.
    """
    n = _n_boosters_from_length_km(L_km, km_per_boost)
    if n == 0:
        return []
    positions = []
    for k in range(1, n + 1):
        d_km = k * km_per_boost
        s = d_km / max(L_km, 1e-9)            # normalized curvilinear abscissa
        s = min(max(s, 0.0), 1.0 - 1e-9)      # keep strictly inside segment
        pt = geom.interpolate(s, normalized=True)
        dx, dy = _tangent_dir_at(geom, s)
        positions.append((pt.x, pt.y, dx, dy))
    return positions

# ---------------------------------------------------------------------------
# BOOSTER PATCHES (symbol)
# ---------------------------------------------------------------------------
def booster_patch(size=40):
    """
    Returns a PathPatch composed of:
      • a circle of radius 1
      • an inscribed triangle pointing right (to be rotated later)
    The patch is centered at (0,0); scaling/rotation/translation is applied later.
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
    """Return a marker path based on the circle+triangle patch."""
    return booster_patch().get_path()

def booster_legend_handle(markersize=8):
    """Return a Line2D handle for legend."""
    return mlines.Line2D([],
                         [],
                         marker=booster_marker(),
                         markersize=markersize,
                         markerfacecolor="none",
                         markeredgecolor="black",
                         linestyle="",
                         label="Boosting station")

# ---------------------------------------------------------------------------
# PIPES (background & overlay)
# ---------------------------------------------------------------------------
def add_pipes(ax, DATA, model, year=2050, w: str="base_utilization",
              km_per_boost: float = 150.0,
              pipe_color="#457bb7",
              alpha=0.7):
    """
    1) Draw all candidate pipes in light gray (background).
    2) Overlay ACTIVE built pipes in a single solid color (blue by default).
       Pressure-related gradients and colorbars are removed.
    3) Draw 'traditional' boosters every `km_per_boost` km from the origin node,
       oriented following the local pipe direction.
    """
    m = model

    # 1) Background in gray
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
        ax.plot(x, y, color="gray", linewidth=0.6, alpha=0.35,
                linestyle=ls, zorder=2)

    # 2) Overlay with results
    t_match = next(t for t in m.T if int(t) == year)

    for p in list(m.P_on) + list(m.P_off):
        is_on = (p in m.P_on)

        # Built? (pipeline activity only; shipping does NOT draw the pipe)
        active = (pyo.value(m.act_on[p, t_match, w]) > 0.5) if is_on else (pyo.value(m.act_off[p, t_match, w]) > 0.5)
        if not active:
            continue

        # Selected diameter (for label)
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
            # if no diameter selected, skip drawing overlay to avoid confusion
            continue

        geom = DATA["pipe_geom"][p]
        L_km = DATA["L"][p]

        # Draw active pipeline in a single color
        x, y = geom.xy
        ax.plot(x, y, color=pipe_color, linewidth=3.0, alpha=0.95, zorder=4)

        # ------------------------------------------------------------------
        # Diameter label (same strategy as before, but pressure-free)
        # ------------------------------------------------------------------
        label = f'⌀ {diam_selected}'
        mid_pt = geom.interpolate(0.5, normalized=True)
        label_centers = []

        offsets = [(15, 0), (0, -12), (-15, 0), (0, 12), (15, 12), (15, -12), (-15, 12), (-15, -12)]

        node_points = [Point(xy) for xy in DATA["coords"].values()]
        active_lines = [geom]

        def overlaps(pt, tol=0.15):
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

        for dx_pt, dy_pt in offsets:
            cand = ax.transData.inverted().transform(
                    ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array([dx_pt, dy_pt]))
            cand_pt = Point(cand)
            if not overlaps(cand_pt):
                chosen_offset = (dx_pt, dy_pt)
                break
        else:
            chosen_offset = offsets[-1]

        label_centers.append(Point(
            ax.transData.inverted().transform(
                ax.transData.transform((mid_pt.x, mid_pt.y)) + np.array(chosen_offset)
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

        # ------------------------------------------------------------------
        # Traditional boosters: every km_per_boost km from origin
        # ------------------------------------------------------------------
        boosters = _traditional_booster_positions(geom, L_km, km_per_boost=km_per_boost)
        for bx, by, dx, dy in boosters:
            angle = np.arctan2(dy, dx)
            patch = booster_patch()
            # rough degree scaling; correct for longitude distortion with cos(lat)
            lat0 = by
            cosphi = max(0.2, np.cos(np.deg2rad(lat0)))  # avoid division by near-zero
            scale_deg = 0.08
            x_scale, y_scale = (scale_deg / cosphi), scale_deg
            trans = (mtrans.Affine2D()
                     .rotate(angle)
                     .scale(x_scale, y_scale)
                     .translate(bx, by)
                     + ax.transData)
            patch.set_transform(trans)
            patch.set_zorder(7)
            patch.set_clip_on(False)
            ax.add_patch(patch)

# ---------------------------------------------------------------------------
# PLOT ENTRYPOINTS ----------------------------------------------------------
# ---------------------------------------------------------------------------
def plot_network(m, DATA, scenario="base_utilization", year=2050,
                 save=True, show=True, theta_override: float | None = None,
                 km_per_boost: float = 150.0):
    """
    Generate the map for (scenario, year).
    Shows only this one on screen; use save_all_plots() to generate all 15 PNGs.
    """
    import rasterio
    dem_path = "merged_srtm.tif"
    with rasterio.open(dem_path) as src:
        scale = 8
        elev = src.read(1, out_shape=(src.height//scale, src.width//scale))
        xmin, ymin, xmax, ymax = src.bounds
        extent = (xmin, xmax, ymin, ymax)

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_relief(ax, elev, extent)

    # gentle vertical crop
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin + 0.5, ymax - 1)

    # nodes and pipes (scenario/year)
    add_nodes(ax, DATA)
    add_pipes(ax, DATA, m, year=year, w=scenario, km_per_boost=km_per_boost)

    # --- Year badge (top-right) ------------------------------------------
    ax.text(
        0.98, 0.98,
        f"{year}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=16,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec="none",
            alpha=0.7
        ),
        zorder=10
    )

    if save:
        ths = _theta_str(m, theta_override)
        subdir = _OUTDIR / f"output_stochastic_traditional_theta_{ths}"
        subdir.mkdir(parents=True, exist_ok=True)
        suf = _suffix_for(scenario)
        outpath = subdir / f"stochastic_traditional_solution_{suf}_theta_{ths}_{int(year)}.png"
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"✔  Mapa guardado en {outpath.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def save_all_plots(m, DATA, theta_override: float | None = None, km_per_boost: float = 150.0):
    """Generate and save 3×5 plots (no display)."""
    years = sorted(int(t) for t in m.T)
    scenarios = list(m.W)
    for w in scenarios:
        for yr in years:
            plot_network(m, DATA, scenario=w, year=yr, save=True, show=False,
                         theta_override=theta_override, km_per_boost=km_per_boost)
