"""
ERA5 mean January temperature map for Spain and Portugal

Reads the previously downloaded ERA5 file (daily means), averages
over all days, masks everything outside Spain and Portugal and
overlays the candidate CO2 pipeline network.

Source:
Copernicus Climate Data Store
ERA5 Reanalysis
"""

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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

INPUT_FILE = "era5_daily_january_europe.nc"
MONTH_NAME = "January"
FIGURE_FILE = "era5_january_heatmap_iberia.png"

NETWORK_FILE = "iberian_co2_network_data.xlsx"
NETWORK_SHEET = "Pipeline candidates"


# ==========================================================
# Load data + temporal mean
# ==========================================================

ds = xr.open_dataset(INPUT_FILE)

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


# ==========================================================
# Finer grid + Spain/Portugal mask
# ==========================================================

lat_new = np.linspace(AREA[0], AREA[2], GRID_SIZE)
lon_new = np.linspace(AREA[1], AREA[3], GRID_SIZE)

temp_grid = temp_mean.interp(latitude=lat_new, longitude=lon_new)

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

temp_grid = temp_grid.where(inside)


# ==========================================================
# First-level administrative regions
# (Spain: autonomous communities, Portugal: districts)
# ==========================================================

admin1_file = shpreader.natural_earth(
    resolution="10m",
    category="cultural",
    name="admin_1_states_provinces"
)

admin1_records = [
    record for record in shpreader.Reader(admin1_file).records()
    if record.attributes["admin"] in COUNTRIES
]

# Natural Earth only holds Spanish provinces -> merge them per autonomous community
spain_regions = {}
for record in admin1_records:
    if record.attributes["admin"] == "Spain":
        spain_regions.setdefault(record.attributes["region"], []).append(record.geometry)

admin1_shapes = [unary_union(geoms) for geoms in spain_regions.values()] + [
    record.geometry for record in admin1_records
    if record.attributes["admin"] == "Portugal"
]


# ==========================================================
# Candidate pipeline network
# ==========================================================

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
# Map
# ==========================================================

plt.figure(figsize=(12, 8))

# Lambert conformal conic projection -> no east-west stretching as with PlateCarree
ax = plt.axes(projection=ccrs.LambertConformal(
    central_longitude=-3.5,
    central_latitude=40,
    standard_parallels=(37, 43)
))
ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())

# Background below, outlines above the temperature field
ax.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0)

heatmap = ax.pcolormesh(
    temp_grid.longitude,
    temp_grid.latitude,
    temp_grid,
    cmap="coolwarm",
    shading="auto",
    transform=ccrs.PlateCarree(),
    zorder=1
)

ax.coastlines(resolution="10m", linewidth=0.8, zorder=2)
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1.0, zorder=2)
ax.add_geometries(
    admin1_shapes,
    crs=ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="black",
    linewidth=0.25,
    zorder=2
)

# Candidate pipeline network (onshore solid, offshore dashed)
for _, pipe in network.iterrows():
    lon, lat = pipe["line"].xy
    ax.plot(
        lon, lat,
        color="black",
        linewidth=0.8,
        linestyle="-" if pipe["Transport method"] == "Onshore" else "--",
        transform=ccrs.PlateCarree(),
        zorder=3
    )

ax.scatter(
    node_lon, node_lat,
    s=30,
    color="black",
    edgecolors="white",
    linewidths=0.4,
    transform=ccrs.PlateCarree(),
    zorder=4
)

# Colorbar exactly as tall as the map (map size is fixed by the projection aspect)
ax.apply_aspect()
map_pos = ax.get_position()
cax = plt.gcf().add_axes([map_pos.x1 + 0.015, map_pos.y0, 0.02, map_pos.height])

cbar = plt.colorbar(heatmap, cax=cax, orientation="vertical")
cbar.set_label("Temperature [°C]", fontsize=16)
cbar.ax.tick_params(labelsize=14)
plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", FIGURE_FILE)
