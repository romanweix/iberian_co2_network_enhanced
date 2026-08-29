import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

# ==========================================================
# Einstellungen
# ==========================================================

INPUT_FILE = "era5_january_mean_last5years.nc"
OUTPUT_FILE_IBERIA = "era5_january_mean_iberia.nc"
FIGURE_FILE_IBERIA = "era5_january_heatmap_iberia.png"

# Bounding Box Iberische Halbinsel [Nord, West, Süd, Ost]
IBERIA_AREA = {
    "lat_max": 44.0,  # Nord
    "lat_min": 35.0,  # Süd
    "lon_min": -10.0,  # West
    "lon_max": 4.5,  # Ost
}

# 1. Datenraster / Pixelauflösung (None = Originalauflösung behalten)
# Beispiel: 100 für ein 100x100 Raster
DATA_GRID_SIZE = 100

# 2. Legende / Temperaturgrenzen (°C)
MIN_TEMP = -2.0  # Untere Grenze (vmin)
MAX_TEMP = 18.0  # Obere Grenze (vmax)

# 3. Kartengitter / Gridlines (Abstand in Grad)
GRID_SPACING_LON = 2.0  # Alle X° Längengrad
GRID_SPACING_LAT = 2.0  # Alle Y° Breitengrad


# ==========================================================
# Daten laden & Zuschneiden
# ==========================================================

print("Lese Daten ein...")
ds = xr.open_dataset(INPUT_FILE)

# Variable ermitteln (xr.Dataset zu xr.DataArray auflösen, falls nötig)
if isinstance(ds, xr.Dataset):
    var_name = list(ds.data_vars)[0]
    temp_grid = ds[var_name]
else:
    temp_grid = ds

# Koordinatennamen dynamisch ermitteln (lat/latitude, lon/longitude)
lat_name = "latitude" if "latitude" in temp_grid.coords else "lat"
lon_name = "longitude" if "longitude" in temp_grid.coords else "lon"

# Zuschneiden über .sel()
lat_slice = (
    slice(IBERIA_AREA["lat_max"], IBERIA_AREA["lat_min"])
    if temp_grid[lat_name][0] > temp_grid[lat_name][-1]
    else slice(IBERIA_AREA["lat_min"], IBERIA_AREA["lat_max"])
)

temp_iberia = temp_grid.sel(
    {
        lat_name: lat_slice,
        lon_name: slice(IBERIA_AREA["lon_min"], IBERIA_AREA["lon_max"]),
    }
)


# ==========================================================
# Optional: Datenraster verändern (Pixel-Interpolation)
# ==========================================================

if DATA_GRID_SIZE is not None:
    print(
        f"Re-interpoliere Daten auf ein {DATA_GRID_SIZE}x{DATA_GRID_SIZE} Raster..."
    )

    lat_new = np.linspace(
        IBERIA_AREA["lat_max"], IBERIA_AREA["lat_min"], DATA_GRID_SIZE
    )
    lon_new = np.linspace(
        IBERIA_AREA["lon_min"], IBERIA_AREA["lon_max"], DATA_GRID_SIZE
    )

    temp_iberia = temp_iberia.interp(
        {lat_name: lat_new, lon_name: lon_new}, method="linear"
    )

# NetCDF speichern
temp_iberia.to_netcdf(OUTPUT_FILE_IBERIA)
print(f"Gespeichert: {OUTPUT_FILE_IBERIA}")


# ==========================================================
# Heatmap & Karte erzeugen
# ==========================================================

print("Erzeuge Karte...")

plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Karten-Ausschnitt setzen
ax.set_extent(
    [
        IBERIA_AREA["lon_min"],
        IBERIA_AREA["lon_max"],
        IBERIA_AREA["lat_min"],
        IBERIA_AREA["lat_max"],
    ],
    crs=ccrs.PlateCarree(),
)

# Heatmap mit festgelegten Limits (vmin/vmax)
heatmap = ax.pcolormesh(
    temp_iberia[lon_name],
    temp_iberia[lat_name],
    temp_iberia,
    cmap="coolwarm",
    vmin=MIN_TEMP,
    vmax=MAX_TEMP,
    shading="auto",
    transform=ccrs.PlateCarree(),
)


# ==========================================================
# Kartenelemente, Grenzen & Gridlines
# ==========================================================

# Land & Ozean Flächen
ax.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0)

# Verwendungsstarke/Dickere Grenzen
ax.coastlines(resolution="10m", linewidth=1.8, color="black", zorder=2)
ax.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    linewidth=1.5,
    edgecolor="black",
    zorder=2,
)

# Koordinatennetz auf der Karte
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.8,
    color="gray",
    alpha=0.6,
    linestyle="--",
    zorder=3,
)

# Beschriftung nur links und unten
gl.top_labels = False
gl.right_labels = False

# Abstände der Gradlinien steuern
gl.xlocator = mticker.MultipleLocator(GRID_SPACING_LON)
gl.ylocator = mticker.MultipleLocator(GRID_SPACING_LAT)


# ==========================================================
# Farbskala & Titel
# ==========================================================

cbar = plt.colorbar(
    heatmap, orientation="vertical", shrink=0.8, pad=0.04, extend="both"
)
cbar.set_label("Temperature [°C]", fontsize=11)

plt.title(
    "Average Temperature January (2021-2025) [°C]",
    fontsize=13,
    pad=12,
)

plt.tight_layout()

# Speichern & Anzeigen
plt.savefig(FIGURE_FILE_IBERIA, dpi=300, bbox_inches="tight")
plt.show()

print(f"Fertig: {FIGURE_FILE_IBERIA}")