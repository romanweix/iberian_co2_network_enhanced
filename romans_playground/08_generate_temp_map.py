"""
ERA5 mean 2 m air temperature maps for Spain and Portugal

Reads the previously downloaded ERA5 files (daily means), averages
over all days, masks everything outside Spain and Portugal and
overlays the candidate CO2 pipeline network.

Two versions are produced per month:
- standard:   PNG for quick inspection
- scientific: publication-ready figure (vector PDF + 600 dpi PNG,
              serif font, graticule labels, legend)

Source:
Copernicus Climate Data Store
ERA5 Reanalysis
"""

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely
from shapely import wkt
from shapely.ops import unary_union


# ==========================================================
# Settings
# ==========================================================

GRID_SIZE = 250

# Iberian Peninsula incl. Balearic Islands (excl. Canary Islands): [N, W, S, E]
AREA = [44.5, -10, 35.5, 5]

COUNTRIES = ["Spain", "Portugal"]

NETWORK_FILE = "iberian_co2_network_data.xlsx"
NETWORK_SHEET = "Pipeline candidates"

# Per month: (ERA5 file, month name, output file stem)
MONTHS = [
    ("era5_daily_january_europe.nc", "January", "era5_january_heatmap_iberia"),
    ("era5_daily_july_europe.nc", "July", "era5_july_heatmap_iberia"),
]

YEARS = "2021–2025"

# Lambert conformal conic projection -> no east-west stretching as with PlateCarree
PROJECTION = ccrs.LambertConformal(
    central_longitude=-3.5,
    central_latitude=40,
    standard_parallels=(37, 43)
)

# Publication style: 180 mm double-column width, serif font
SCIENTIFIC_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.3",
    "pdf.fonttype": 42,
}
SCIENTIFIC_WIDTH_IN = 180 / 25.4


# ==========================================================
# Spain/Portugal mask + pipeline network (shared by all months)
# ==========================================================

lat_new = np.linspace(AREA[0], AREA[2], GRID_SIZE)
lon_new = np.linspace(AREA[1], AREA[3], GRID_SIZE)

shapefile = shpreader.natural_earth(
    resolution="10m",
    category="cultural",
    name="admin_0_countries"
)

iberia_shape = unary_union([
    country.geometry
    for country in shpreader.Reader(shapefile).records()
    if country.attributes["NAME_EN"] in COUNTRIES
])

lon_2d, lat_2d = np.meshgrid(lon_new, lat_new)
inside = shapely.contains_xy(iberia_shape, lon_2d, lat_2d)

network = pd.read_excel(NETWORK_FILE, sheet_name=NETWORK_SHEET)
network["line"] = network["geometry"].apply(wkt.loads)

# Nodes = start and end points of the candidate pipelines
nodes = {
    point
    for line in network["line"]
    for point in (line.coords[0], line.coords[-1])
}
node_lon, node_lat = zip(*nodes)


# ==========================================================
# Plot function
# ==========================================================

def plot_map(temp_grid, month_name, scientific=False):

    if scientific:
        fig = plt.figure(figsize=(SCIENTIFIC_WIDTH_IN, SCIENTIFIC_WIDTH_IN * 0.72))
        line_width, node_size = 0.6, 6
    else:
        fig = plt.figure(figsize=(12, 8))
        line_width, node_size = 0.8, 12

    ax = plt.axes(projection=PROJECTION)
    ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())

    # Background below, outlines above the temperature field
    ax.add_feature(cfeature.LAND, facecolor="#eeeeee", rasterized=scientific, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", rasterized=scientific, zorder=0)

    heatmap = ax.pcolormesh(
        temp_grid.longitude,
        temp_grid.latitude,
        temp_grid,
        cmap="coolwarm",
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=scientific,
        zorder=1
    )

    ax.coastlines(resolution="10m", linewidth=line_width, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=line_width * 0.6, zorder=2)

    # Candidate pipeline network (onshore solid, offshore dashed)
    for _, pipe in network.iterrows():
        lon, lat = pipe["line"].xy
        ax.plot(
            lon, lat,
            color="black",
            linewidth=line_width,
            linestyle="-" if pipe["Transport method"] == "Onshore" else "--",
            transform=ccrs.PlateCarree(),
            zorder=3
        )

    ax.scatter(
        node_lon, node_lat,
        s=node_size,
        color="black",
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    cbar = plt.colorbar(heatmap, orientation="vertical", shrink=0.7, pad=0.03)

    if scientific:
        cbar.set_label(f"Mean {month_name} 2 m air temperature, {YEARS} (°C)")
        cbar.outline.set_linewidth(0.6)

        gl = ax.gridlines(
            draw_labels=True,
            x_inline=False,
            y_inline=False,
            linewidth=0.3,
            color="0.5",
            linestyle=":",
            rotate_labels=False,
            zorder=5
        )
        gl.xlocator = mticker.FixedLocator(range(-10, 6, 2))
        gl.ylocator = mticker.FixedLocator(range(36, 45, 2))
        gl.top_labels = False
        gl.right_labels = False

        ax.legend(
            handles=[
                Line2D([], [], color="black", linewidth=line_width, label="Onshore candidate pipeline"),
                Line2D([], [], color="black", linewidth=line_width, linestyle="--", label="Offshore candidate pipeline"),
                Line2D([], [], color="black", marker="o", markersize=3, linestyle="", label="Network node"),
            ],
            loc="lower right"
        )
    else:
        cbar.set_label(f"Mean {month_name} temperature [°C]")

    return fig


# ==========================================================
# Maps per month
# ==========================================================

for input_file, month_name, output_stem in MONTHS:

    ds = xr.open_dataset(input_file)

    # ERA5 variable name depends on the CDS version
    temp = ds["t2m"] if "t2m" in ds else ds["2m_temperature"]
    temp = temp - 273.15

    time_dim = "valid_time" if "valid_time" in temp.dims else "time"

    # ERA5 latitudes are usually descending (north -> south)
    if temp.latitude[0] > temp.latitude[-1]:
        lat_slice = slice(AREA[0], AREA[2])
    else:
        lat_slice = slice(AREA[2], AREA[0])

    temp_mean = temp.sel(
        latitude=lat_slice,
        longitude=slice(AREA[1], AREA[3])
    ).mean(dim=time_dim)

    temp_grid = temp_mean.interp(latitude=lat_new, longitude=lon_new)
    temp_grid = temp_grid.where(inside)

    # Standard version
    fig = plot_map(temp_grid, month_name)
    fig.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")
    print("Saved:", f"{output_stem}.png")

    # Scientific version
    with plt.rc_context(SCIENTIFIC_STYLE):
        fig = plot_map(temp_grid, month_name, scientific=True)
        for ext, dpi in (("pdf", 300), ("png", 600)):
            fig.savefig(f"{output_stem}_scientific.{ext}", dpi=dpi, bbox_inches="tight")
            print("Saved:", f"{output_stem}_scientific.{ext}")

plt.show()
