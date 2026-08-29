import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString

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

# 3. Verwaltungsgrenzen (Comunidades Autónomas / Distritos) statt Längen-
# /Breitengrad-Gitter; auf diese Länder beschränkt (ISO-A3)
ADMIN_BOUNDARY_COUNTRIES = ["ESP", "PRT"]

# 4. Pipeline-Trassen aus der Kandidatenliste einzeichnen (None = keine)
EXCEL = "iberian_co2_network_data.xlsx"
SHEETNAME = "Pipeline candidates"
PIPE_ID = "1_a"
DEM = "merged_srtm_roman.tif"  # für Höhenprofil von PIPE_ID
PROFILE_STEP = 500  # Meter, Schrittweite entlang der Trasse für die Profile


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
# Pipeline-Trassen aus Excel einlesen
# ==========================================================

# Jede physische Trasse liegt zweimal vor (z.B. "1_a"/"1_b" mit exakt
# umgekehrter Geometrie, für Hin-/Rückrichtung) - für die Kartendarstellung
# reicht die "_a"-Variante. "Stage" == "First"/"Second" wird als primäre
# bzw. sekundäre Leitung interpretiert (durchgezogen bzw. gestrichelt).
pipeline_line = None
other_pipelines_a = pd.DataFrame()

if PIPE_ID is not None:
    print(f"Lese Pipeline-Trassen aus {EXCEL} ...")
    pipeline_df = pd.read_excel(EXCEL, sheet_name=SHEETNAME)
    pipelines_a = pipeline_df[
        pipeline_df["Pipeline identifier"].str.endswith("_a")
    ]

    pipeline_row = pipeline_df[pipeline_df["Pipeline identifier"] == PIPE_ID]
    if pipeline_row.empty:
        print(
            f"Warnung: Pipeline '{PIPE_ID}' nicht in {EXCEL} (Sheet "
            f"'{SHEETNAME}') gefunden - es wird keine Trasse eingezeichnet."
        )
    else:
        # Die Koordinaten in der Excel-Tabelle sind WGS84 (wie ax.set_extent
        # über ccrs.PlateCarree()), daher direkt ohne Umprojektion nutzbar.
        pipeline_line = wkt.loads(pipeline_row.iloc[0]["geometry"])

    other_pipelines_a = pipelines_a[
        pipelines_a["Pipeline identifier"] != PIPE_ID
    ]


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

# Verwaltungsgrenzen (Comunidades Autónomas / Distritos) - dünn und grau,
# damit sie sich klar von der (dickeren, schwarzen) Landesgrenze absetzen.
# Aus dem Polygon-Datensatz statt admin_1_states_provinces_lines, da
# Länder-Attribute (ADM0_A3) dort pro Segment oft leer sind (Segmente
# liegen zwischen zwei Einheiten und tragen nur NAME_L/NAME_R).
admin1_path = shpreader.natural_earth(
    resolution="10m", category="cultural", name="admin_1_states_provinces"
)
admin1_geoms = [
    rec.geometry
    for rec in shpreader.Reader(admin1_path).records()
    if rec.attributes.get("adm0_a3") in ADMIN_BOUNDARY_COUNTRIES
]
ax.add_geometries(
    admin1_geoms,
    crs=ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="0.45",
    linewidth=0.4,
    zorder=1.5,
    rasterized=True,
)

# Küsten- & Landesgrenzen (kräftiger als die Verwaltungsgrenzen, damit die
# Hierarchie Land > Region klar erkennbar bleibt)
ax.coastlines(resolution="10m", linewidth=0.8, color="black", zorder=2, rasterized=True)
ax.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    linewidth=0.9,
    edgecolor="black",
    zorder=2,
    rasterized=True,
)

# Alle übrigen "_a"-Trassen dünn im Hintergrund (durchgezogen = primär/
# "First", gestrichelt = sekundär/"Second"), PIPE_ID bleibt für den
# hervorgehobenen Plot direkt danach ausgespart. Knoten (Start-/Endpunkte)
# aller Kanten werden gesammelt und am Schluss einheitlich eingezeichnet -
# PIPE_ID bekommt dabei bewusst keine eigene Markeroptik mehr.
node_coords = set()
for _, row in other_pipelines_a.iterrows():
    try:
        route = wkt.loads(row["geometry"])
    except Exception:
        continue
    route_lons, route_lats = route.xy
    node_coords.update(route.coords)
    is_primary = row["Stage"] == "First"
    ax.plot(
        route_lons,
        route_lats,
        color="0.25",
        linewidth=0.8,
        linestyle="-" if is_primary else "--",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=3.5,
    )

# Hervorgehobene Trasse PIPE_ID (weiße Kontur für Lesbarkeit über hellen wie
# dunklen Heatmap-Bereichen, bleibt Vektorgrafik statt rasterisiert für
# scharfe Kanten in der PDF-Fassung)
if pipeline_line is not None:
    node_coords.update(pipeline_line.coords)
    pipe_lons, pipe_lats = pipeline_line.xy
    ax.plot(
        pipe_lons,
        pipe_lats,
        color="black",
        linewidth=2.0,
        transform=ccrs.PlateCarree(),
        zorder=4,
        path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
    )

# Knoten des Graphen (Kantenendpunkte) - dezent, aber durch den weißen
# Rand auf jedem Untergrund gut sichtbar; einheitlich für PIPE_ID wie für
# alle anderen Trassen, keine Sonderdarstellung.
if node_coords:
    node_lons, node_lats = zip(*node_coords)
    ax.scatter(
        node_lons,
        node_lats,
        s=16,
        color="0.2",
        edgecolor="white",
        linewidth=0.6,
        transform=ccrs.PlateCarree(),
        zorder=4.5,
    )

# Bewusst kein Längen-/Breitengrad-Gitter (siehe ADMIN_BOUNDARY_COUNTRIES
# oben) - ohne gridlines(draw_labels=True) zeichnet die GeoAxes auch keine
# Tick-Beschriftung, das Kartenbild bleibt entsprechend unbeschriftet.


# ==========================================================
# Farbskala & Titel
# ==========================================================

cbar = plt.colorbar(
    heatmap, orientation="vertical", shrink=0.85, pad=0.04, extend="both"
)
cbar.set_label("temperature [$^{\\circ}$C]", fontsize=10.5)
cbar.ax.tick_params(labelsize=9)

# Hinweis: Der Titel wird bewusst über fig.suptitle() statt ax.set_title()
# gesetzt. Mit aktivierten Gridliner-Labels (draw_labels=True, mittlerweile
# entfernt) löste cartopys interne Titel-Positionierung
# (_update_title_position) auf GeoAxes einen Fehler aus (mit
# plt.tight_layout(): eine shapely-Exception beim Rendern) bzw. verschob
# den Titel unsichtbar aus der Figur heraus (ohne tight_layout).
# fig.suptitle() umgeht diese GeoAxes-spezifische Positionierungslogik
# vollständig und bleibt daher auch ohne Gridliner die robustere Wahl.
# Aus demselben Grund wird auch kein bbox_inches="tight" beim Speichern
# verwendet (schneidet die Karte auf die Colorbar zusammen) – die Ränder
# werden stattdessen über fig.subplots_adjust() oben gesetzt.
title = "Mean January 2 m air temperature, Iberian Peninsula (2021–2025)"
if pipeline_line is not None:
    title += f", with pipeline {PIPE_ID} route"

fig.suptitle(
    title,
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

print(f"Fertig: {FIGURE_FILE_IBERIA}, {figure_path}.png, {figure_path}.pdf")


# ==========================================================
# Höhen- & Temperaturprofil entlang PIPE_ID
# ==========================================================

if pipeline_line is not None:
    print(f"Berechne Höhen-/Temperaturprofil für '{PIPE_ID}' ...")

    dem_src = rasterio.open(DEM)
    to_dem_crs = Transformer.from_crs("EPSG:4326", dem_src.crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(dem_src.crs, "EPSG:4326", always_xy=True)

    # Für gleichmäßige Schrittweite in Metern wird die Trasse in das
    # (metrische) DEM-Koordinatensystem projiziert, entlang derer
    # interpoliert und je Punkt zurück nach WGS84 transformiert (für die
    # Temperatur, die auf einem Lat/Lon-Gitter vorliegt).
    coords_dem = [to_dem_crs.transform(x, y) for x, y in pipeline_line.coords]
    line_dem = LineString(coords_dem)
    length_m = line_dem.length

    prof_distances_m = np.arange(0, length_m + PROFILE_STEP, PROFILE_STEP)
    prof_elevations = []
    prof_temperatures = []

    for d in prof_distances_m:
        point = line_dem.interpolate(d)

        elevation = next(dem_src.sample([(point.x, point.y)]))[0]
        prof_elevations.append(float(elevation))

        lon, lat = to_wgs84.transform(point.x, point.y)
        T = temp_grid.interp({lat_name: lat, lon_name: lon}, method="linear")
        prof_temperatures.append(float(T.values))

    dem_src.close()

    prof_distances_km = prof_distances_m / 1000
    profile_base = os.path.join(FIGURE_DIR, f"pipeline_{PIPE_ID}")

    # --- Höhenprofil (Topologie) ---
    fig_topo, ax_topo = plt.subplots(figsize=(7.5, 3.2))
    ax_topo.fill_between(prof_distances_km, prof_elevations, color="#8c6a4a", alpha=0.30, lw=0)
    ax_topo.plot(prof_distances_km, prof_elevations, color="#6b4d30", lw=1.1)
    ax_topo.set_xlim(prof_distances_km[0], prof_distances_km[-1])
    ax_topo.set_ylim(bottom=0)
    ax_topo.set_xlabel("distance along pipeline [km]")
    ax_topo.set_ylabel("elevation [m a.s.l.]")
    ax_topo.grid(True, alpha=0.3)
    ax_topo.set_title(
        f"Pipeline {PIPE_ID} — topology profile", fontsize=12, fontweight="bold"
    )
    fig_topo.tight_layout()
    fig_topo.savefig(f"{profile_base}_topology_profile.png", dpi=300)
    fig_topo.savefig(f"{profile_base}_topology_profile.pdf")

    # --- Temperaturprofil ---
    fig_temp, ax_temp = plt.subplots(figsize=(7.5, 3.2))
    ax_temp.plot(prof_distances_km, prof_temperatures, color="firebrick", lw=1.3)
    ax_temp.axhline(0, color="0.5", lw=0.8, ls="--")
    ax_temp.set_xlim(prof_distances_km[0], prof_distances_km[-1])
    ax_temp.set_xlabel("distance along pipeline [km]")
    ax_temp.set_ylabel("temperature [$^{\\circ}$C]")
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_title(
        f"Pipeline {PIPE_ID} — mean January ambient temperature profile",
        fontsize=12,
        fontweight="bold",
    )
    fig_temp.tight_layout()
    fig_temp.savefig(f"{profile_base}_temperature_profile.png", dpi=300)
    fig_temp.savefig(f"{profile_base}_temperature_profile.pdf")

    print(
        f"Fertig: {profile_base}_topology_profile.png/.pdf, "
        f"{profile_base}_temperature_profile.png/.pdf"
    )

plt.show()