"""
ERA5 Spanien + Portugal Juli Temperaturkarte

Liest die bereits heruntergeladene ERA5-Datei (Tagesmittel),
mittelt über alle Tage, maskiert alles außer Spanien und Portugal
und speichert eine Heatmap.

Quelle:
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
# Einstellungen
# ==========================================================

GRID_SIZE = 250

# Iberische Halbinsel inkl. Balearen (ohne Kanaren): [Nord, West, Süd, Ost]
AREA = [44.5, -10, 35.5, 5]

COUNTRIES = ["Spain", "Portugal"]

INPUT_FILE = "era5_daily_july_europe.nc"
NETWORK_FILE = "iberian_co2_network_data.xlsx"
NETWORK_SHEET = "Pipeline candidates"
FIGURE_FILE = "era5_july_heatmap_iberia.png"

YEARS = "2021–2025"


# ==========================================================
# Daten laden + Mittelwert
# ==========================================================

ds = xr.open_dataset(INPUT_FILE)

# ERA5 Variable kann je nach CDS-Version anders heißen
temp = ds["t2m"] if "t2m" in ds else ds["2m_temperature"]
temp = temp - 273.15

time_dim = "valid_time" if "valid_time" in temp.dims else "time"

# ERA5 liefert Breitengrade meist absteigend (Nord -> Süd)
if temp.latitude[0] > temp.latitude[-1]:
    lat_slice = slice(AREA[0], AREA[2])
else:
    lat_slice = slice(AREA[2], AREA[0])

temp_mean = temp.sel(
    latitude=lat_slice,
    longitude=slice(AREA[1], AREA[3])
).mean(dim=time_dim)


# ==========================================================
# Feineres Raster + Maske Spanien/Portugal
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
# Heatmap
# ==========================================================

plt.figure(figsize=(12, 8))

# Winkeltreue Projektion -> keine Ost-West-Streckung wie bei PlateCarree
ax = plt.axes(projection=ccrs.LambertConformal(
    central_longitude=-3.5,
    central_latitude=40,
    standard_parallels=(37, 43)
))
ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())

# Hintergrund unter, Linien über der Heatmap
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
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, zorder=2)

# Kandidaten-Pipelinenetz (Onshore durchgezogen, Offshore gestrichelt)
network = pd.read_excel(NETWORK_FILE, sheet_name=NETWORK_SHEET)

for _, pipe in network.iterrows():
    lon, lat = wkt.loads(pipe["geometry"]).xy
    ax.plot(
        lon, lat,
        color="black",
        linewidth=0.8,
        linestyle="-" if pipe["Transport method"] == "Onshore" else "--",
        transform=ccrs.PlateCarree(),
        zorder=3
    )

cbar = plt.colorbar(heatmap, orientation="vertical", shrink=0.7)
cbar.set_label("Mittlere Julitemperatur [°C]")

plt.title(
    "Spanien & Portugal: mittlere Julitemperatur\n"
    f"ERA5 Mittel {YEARS}"
)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
plt.show()

print("Fertig:", FIGURE_FILE)
