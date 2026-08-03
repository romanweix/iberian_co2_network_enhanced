"""
ERA5 Europa Januar Temperaturkarte
Optimierung:
- Tagesmittelwerte statt Stundenwerte
- direkter Mittelwert über Januar
- 50x50 Raster
- Ausgabe als NetCDF + Heatmap

Quelle:
Copernicus Climate Data Store
ERA5 Reanalysis
"""

import os
from datetime import datetime

import cdsapi
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==========================================================
# Einstellungen
# ==========================================================

GRID_SIZE = 50

# Europa
AREA = [
    72,    # Nord
    -25,   # West
    34,    # Süd
    45     # Ost
]

DOWNLOAD_FILE = "era5_daily_january_europe.nc"
OUTPUT_FILE = "era5_january_mean_last5years.nc"
FIGURE_FILE = "era5_january_heatmap.png"


# ==========================================================
# Jahre automatisch bestimmen
# ==========================================================

years = [
    "2021",
    "2022",
    "2023",
    "2024",
    "2025"
]

print("Verwendete Januar-Monate:")
print(years)


# ==========================================================
# ERA5 Tagesdaten herunterladen
# ==========================================================

if not os.path.exists(DOWNLOAD_FILE):

    print("Starte ERA5 Download...")

    cds = cdsapi.Client()

    cds.retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature"
            ],
            "year": years,
            "month": "01",
            "day": [
                f"{d:02d}"
                for d in range(1, 32)
            ],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": AREA,
            # weniger Daten, für 50x50 Karte ausreichend = 1.0, 1.0
            # 25x25 = 0.5, 0.5
            "grid": [
                0.5,
                0.5
            ],

            "data_format": "netcdf"
        },

        DOWNLOAD_FILE
    )

    print("Download fertig.")

else:
    print("Nutze vorhandene Datei:", DOWNLOAD_FILE)



# ==========================================================
# Daten laden
# ==========================================================

print("Lese Daten...")

ds = xr.open_dataset(DOWNLOAD_FILE)


# ERA5 Variable kann je nach CDS-Version anders heißen
if "2m_temperature" in ds:
    temp = ds["2m_temperature"]

elif "t2m" in ds:
    temp = ds["t2m"]

else:
    raise Exception(
        f"Temperaturvariable nicht gefunden: {list(ds.data_vars)}"
    )


# Kelvin -> Celsius

temp = temp - 273.15


# ==========================================================
# Mittelwert über alle Tage und Jahre
# ==========================================================

print("Berechne Mittelwert...")

temp_mean = temp.mean(
    dim="time"
)


# ==========================================================
# Auf 50x50 Raster bringen
# ==========================================================

print("Erzeuge 50x50 Raster...")

lat_new = np.linspace(
    float(temp_mean.latitude.max()),
    float(temp_mean.latitude.min()),
    GRID_SIZE
)

lon_new = np.linspace(
    float(temp_mean.longitude.min()),
    float(temp_mean.longitude.max()),
    GRID_SIZE
)


temp_grid = temp_mean.interp(
    latitude=lat_new,
    longitude=lon_new
)


# ==========================================================
# Ergebnis speichern
# ==========================================================

temp_grid.to_netcdf(
    OUTPUT_FILE
)

print(
    "Gespeichert:",
    OUTPUT_FILE
)



# ==========================================================
# Heatmap Europa
# ==========================================================

print("Erzeuge Karte...")


plt.figure(
    figsize=(12, 8)
)

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [-25, 45, 34, 72]
)


heatmap = ax.pcolormesh(
    temp_grid.longitude,
    temp_grid.latitude,
    temp_grid,

    cmap="coolwarm",

    shading="auto",

    transform=ccrs.PlateCarree()
)


# Kartenelemente

ax.add_feature(
    cfeature.LAND,
    facecolor="#eeeeee"
)

ax.add_feature(
    cfeature.OCEAN,
    facecolor="#dbeeff"
)

ax.coastlines(
    resolution="50m",
    linewidth=0.8
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5
)


# Farbskala

cbar = plt.colorbar(
    heatmap,
    orientation="vertical",
    shrink=0.7
)

cbar.set_label(
    "Mittlere Januartemperatur [°C]"
)


plt.title(
    "Europa: mittlere Januartemperatur\n"
    f"ERA5 Mittel {years[-1]}–{years[0]}"
)


plt.tight_layout()


plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "Fertig:",
    FIGURE_FILE
)