import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely import wkt
from shapely.geometry import LineString

import rasterio
from pyproj import Transformer


# ==========================================================
# Einstellungen
# ==========================================================

EXCEL = "iberian_co2_network_data.xlsx"
DEM = "merged_srtm.tif"

STEP = 500  # Meter


# ==========================================================
# Excel einlesen
# ==========================================================

df = pd.read_excel(EXCEL)

# erste Leitung
geometry = df.loc[1, "geometry"]

line = wkt.loads(geometry)

print(line)
print(line.length)


# ==========================================================
# DEM öffnen
# ==========================================================

src = rasterio.open(DEM)

print(src.crs)

# Falls DEM in EPSG:3035 vorliegt
transformer = Transformer.from_crs(
    "EPSG:4326",
    src.crs,
    always_xy=True,
)


# ==========================================================
# Punkte entlang der Leitung
# ==========================================================

# Die Koordinaten im Excel sind WGS84.
coords = list(line.coords)

coords_3035 = [
    transformer.transform(x, y)
    for x, y in coords
]

line_3035 = LineString(coords_3035)

length = line_3035.length

print(f"Länge = {length/1000:.2f} km")

import xarray as xr

# ==========================================================
# Temperaturkarte laden
# ==========================================================

temp_ds = xr.open_dataset("era5_january_mean_last5years.nc")

# Variable automatisch finden
temp_var = list(temp_ds.data_vars)[0]
temperature = temp_ds[temp_var]

print("Temperaturvariable:", temp_var)

# ==========================================================
# Höhen- und Temperaturprofil
# ==========================================================

distances = np.arange(0, length + STEP, STEP)

elevations = []
temperatures = []

# Rücktransformation nach WGS84
to_wgs84 = Transformer.from_crs(
    src.crs,
    "EPSG:4326",
    always_xy=True
)

for d in distances:

    p = line_3035.interpolate(d)

    # -------------------------
    # Höhe
    # -------------------------

    elevation = next(src.sample([(p.x, p.y)]))[0]
    elevations.append(float(elevation))

    # -------------------------
    # Temperatur
    # -------------------------

    lon, lat = to_wgs84.transform(p.x, p.y)

    T = temperature.interp(
        latitude=lat,
        longitude=lon,
        method="linear"
    )

    temperatures.append(float(T.values))


src.close()


# ==========================================================
# Erstelle DataFrame
# ==========================================================
profile = pd.DataFrame({
    "distance_m": distances,
    "distance_km": distances / 1000,
    "elevation_m": elevations,
    "temperature_degC": temperatures
})

print(profile.head())

profile.to_csv(
    "pipeline_profile.csv",
    index=False
)

# ==========================================================
# Plot
# ==========================================================

fig, ax1 = plt.subplots(figsize=(14,5))

# Höhenprofil
ax1.plot(
    profile.distance_km,
    profile.elevation_m,
    color="tab:brown",
    lw=2,
    label="Elevation"
)

ax1.set_xlabel("Entfernung [km]")
ax1.set_ylabel("Höhe [m]", color="tab:brown")
ax1.tick_params(axis="y", labelcolor="tab:brown")
ax1.grid(True)

# Temperatur
ax2 = ax1.twinx()

ax2.plot(
    profile.distance_km,
    profile.temperature_degC,
    color="tab:blue",
    lw=2,
    label="Temperature"
)

ax2.set_ylabel("Temperatur [°C]", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("Pipelineprofil: Höhe und mittlere Januartemperatur")

plt.tight_layout()
plt.show()