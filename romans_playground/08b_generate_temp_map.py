import os

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
FIGURE_DIR = "figures"
FIGURE_BASENAME = "era5_january_temperature_iberia"

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

    # Für die Interpolation mit Puffer aus dem UNGESCHNITTENEN Grid neu
    # selektieren: Das native ERA5-Gitter (~0.28°) liegt selten exakt auf
    # den gewünschten Kartengrenzen, wodurch temp_iberia (oben exakt auf
    # IBERIA_AREA zugeschnitten) an den äußersten Rand-Zielpunkten fehlt.
    # xarray.interp() extrapoliert das dann als NaN, was pcolormesh
    # transparent zeichnet - sichtbar als heller Rand am Kartenrand, durch
    # den der Ozean-/Land-Hintergrund durchscheint. Der Puffer stellt
    # sicher, dass die Zielpunkte immer innerhalb der Quelldaten liegen
    # (echte Interpolation statt Extrapolation).
    SEL_BUFFER_DEG = 1.0
    lat_slice_buffered = (
        slice(
            IBERIA_AREA["lat_max"] + SEL_BUFFER_DEG,
            IBERIA_AREA["lat_min"] - SEL_BUFFER_DEG,
        )
        if temp_grid[lat_name][0] > temp_grid[lat_name][-1]
        else slice(
            IBERIA_AREA["lat_min"] - SEL_BUFFER_DEG,
            IBERIA_AREA["lat_max"] + SEL_BUFFER_DEG,
        )
    )
    source_for_interp = temp_grid.sel(
        {
            lat_name: lat_slice_buffered,
            lon_name: slice(
                IBERIA_AREA["lon_min"] - SEL_BUFFER_DEG,
                IBERIA_AREA["lon_max"] + SEL_BUFFER_DEG,
            ),
        }
    )

    temp_iberia = source_for_interp.interp(
        {lat_name: lat_new, lon_name: lon_new}, method="linear"
    )

# NetCDF speichern
temp_iberia.to_netcdf(OUTPUT_FILE_IBERIA)
print(f"Gespeichert: {OUTPUT_FILE_IBERIA}")


# ==========================================================
# Heatmap & Karte erzeugen
# ==========================================================

print("Erzeuge Karte...")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
})

fig = plt.figure(figsize=(9, 5.4))
fig.subplots_adjust(left=0.08, right=0.87, top=0.88, bottom=0.10)
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
    rasterized=True,  # haelt die Vektor-PDF klein (sonst eine Fläche pro Gitterzelle)
)


# ==========================================================
# Kartenelemente, Grenzen & Gridlines
# ==========================================================

# Land & Ozean Flächen
# rasterized=True: Natural-Earth-Geometrien werden von cartopy global (nicht
# vorab auf den Kartenausschnitt zugeschnitten) geladen; ohne Rasterung
# landen alle Pfadpunkte unsichtbar mit im Vektor-PDF (~4 MB statt ~15 KB
# allein für LAND) - hier bewusst rasterisiert, während Gridlines und
# Beschriftungen Vektorgrafik bleiben.
ax.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0, rasterized=True)
ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0, rasterized=True)

# Küsten- & Landesgrenzen (fein, druckgerecht statt bildschirmtauglich-dick)
ax.coastlines(resolution="10m", linewidth=0.8, color="black", zorder=2, rasterized=True)
ax.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    linewidth=0.6,
    edgecolor="black",
    zorder=2,
    rasterized=True,
)

# Koordinatennetz auf der Karte
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="--",
    zorder=3,
)

# Beschriftung nur links und unten
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

# Abstände der Gradlinien steuern
gl.xlocator = mticker.MultipleLocator(GRID_SPACING_LON)
gl.ylocator = mticker.MultipleLocator(GRID_SPACING_LAT)


# ==========================================================
# Farbskala & Titel
# ==========================================================

cbar = plt.colorbar(
    heatmap, orientation="vertical", shrink=0.85, pad=0.04, extend="both"
)
cbar.set_label("temperature [$^{\\circ}$C]", fontsize=10.5)
cbar.ax.tick_params(labelsize=9)

# Hinweis: Der Titel wird bewusst über fig.suptitle() statt ax.set_title()
# gesetzt. Mit aktivierten Gridliner-Labels (draw_labels=True) löst
# cartopys interne Titel-Positionierung (_update_title_position) auf
# GeoAxes einen Fehler aus (mit plt.tight_layout(): eine shapely-Exception
# beim Rendern) bzw. verschiebt den Titel unsichtbar aus der Figur heraus
# (ohne tight_layout). fig.suptitle() umgeht diese GeoAxes-spezifische
# Positionierungslogik vollständig. Aus demselben Grund wird auch kein
# bbox_inches="tight" beim Speichern verwendet (schneidet die Karte auf
# die Colorbar zusammen) – die Ränder werden stattdessen über
# fig.subplots_adjust() oben gesetzt.
fig.suptitle(
    "Mean January 2 m air temperature, Iberian Peninsula (2021–2025)",
    fontsize=12,
    fontweight="bold",
    y=0.96,
)

# --- Speichern: Originalpfad (Kompatibilität) + druckreife PNG/PDF-Fassung ---
plt.savefig(FIGURE_FILE_IBERIA, dpi=300)

os.makedirs(FIGURE_DIR, exist_ok=True)
figure_path = os.path.join(FIGURE_DIR, FIGURE_BASENAME)
plt.savefig(f"{figure_path}.png", dpi=300)
plt.savefig(f"{figure_path}.pdf")

plt.show()

print(f"Fertig: {FIGURE_FILE_IBERIA}, {figure_path}.png, {figure_path}.pdf")