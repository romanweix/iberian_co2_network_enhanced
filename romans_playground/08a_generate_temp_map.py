import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr

# ==========================================================
# Einstellungen für Iberische Halbinsel
# ==========================================================

INPUT_FILE = "era5_january_mean_last5years.nc"
OUTPUT_FILE_IBERIA = "era5_january_mean_iberia.nc"
FIGURE_FILE_IBERIA = "era5_january_heatmap_iberia.png"

# Bounding Box Iberische Halbinsel [Nord, West, Süd, Ost]
# Etwas Rand gelassen, um die gesamte Halbinsel inkl. Balearen gut abzubilden
IBERIA_AREA = {
    "lat_max": 44.0,  # Nord
    "lat_min": 35.0,  # Süd
    "lon_min": -10.0,  # West
    "lon_max": 4.5,  # Ost
}

# ==========================================================
# Daten laden und auf Iberien zuschneiden
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

# Zuschneiden über .sel() (Slice verarbeitet sowohl auf- als auch absteigende Breitengrade)
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

# Zugeschnittene NetCDF-Datei speichern
temp_iberia.to_netcdf(OUTPUT_FILE_IBERIA)
print(f"NetCDF für Iberien gespeichert: {OUTPUT_FILE_IBERIA}")

# ==========================================================
# Heatmap Iberische Halbinsel
# ==========================================================

print("Erzeuge Karte für die Iberische Halbinsel...")

plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Ausschnitt auf der Karte festlegen
ax.set_extent(
    [
        IBERIA_AREA["lon_min"],
        IBERIA_AREA["lon_max"],
        IBERIA_AREA["lat_min"],
        IBERIA_AREA["lat_max"],
    ],
    crs=ccrs.PlateCarree(),
)

heatmap = ax.pcolormesh(
    temp_iberia[lon_name],
    temp_iberia[lat_name],
    temp_iberia,
    cmap="coolwarm",
    shading="auto",
    transform=ccrs.PlateCarree(),
)

# Kartenelemente & Dickere Grenzen
ax.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0)

# Küstenlinien und Grenzen mit erhöhter Liniendicke (linewidth=1.8 / 1.5)
ax.coastlines(resolution="10m", linewidth=1.8, color="black", zorder=2)
ax.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    linewidth=1.5,
    edgecolor="black",
    zorder=2,
)

# Farbskala
cbar = plt.colorbar(heatmap, orientation="vertical", shrink=0.8, pad=0.03)
cbar.set_label("Mittlere Januartemperatur [°C]", fontsize=11)

plt.title(
    "Iberische Halbinsel: Mittlere Januartemperatur (ERA5)",
    fontsize=13,
    pad=12,
)

plt.tight_layout()
plt.savefig(FIGURE_FILE_IBERIA, dpi=300, bbox_inches="tight")
plt.show()

print(f"Fertig: {FIGURE_FILE_IBERIA}")