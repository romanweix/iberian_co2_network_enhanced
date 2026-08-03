#!/usr/bin/env python3
"""
Download Copernicus DEM GLO-30 über OpenTopography,
automatisches Aufteilen in Teilgebiete,
Mosaik,
Reprojektion nach EPSG:3035,
Resampling auf 500 m.

Autor: ChatGPT
"""

from pathlib import Path
import math
import os
import requests
from tqdm import tqdm

import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


# ===========================================================
# Einstellungen
# ===========================================================

API_KEY = "288219a47ae7e0dfd26f824a81723a0d"

# Iberische Halbinsel
WEST = -9.60
EAST = 3.40
SOUTH = 35.90
NORTH = 43.90

# Kachelgröße in Grad
# 2° × 2° ist konservativ und bleibt deutlich unter dem API-Limit.
STEP = 2.0

TARGET_RESOLUTION = 500  # Meter
TARGET_CRS = "EPSG:3035"

TEMP = Path("tmp_dem")
TEMP.mkdir(exist_ok=True)

MERGED = TEMP / "merged_30m.tif"
OUTPUT = "iberia_dem_500m.tif"


# ===========================================================
# Downloadfunktion
# ===========================================================

def download_tile(w, e, s, n, outfile):

    url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": "COP30",
        "south": s,
        "north": n,
        "west": w,
        "east": e,
        "outputFormat": "GTiff",
        "API_Key": API_KEY,
    }

    print(f"Download {outfile.name}")

    r = requests.get(url, params=params, stream=True)

    if r.status_code != 200:
        raise RuntimeError(r.text)

    total = int(r.headers.get("content-length", 0))

    with open(outfile, "wb") as f:

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            leave=False
        ) as pbar:

            for chunk in r.iter_content(1024 * 1024):

                if chunk:

                    f.write(chunk)
                    pbar.update(len(chunk))


# ===========================================================
# Download aller Teilgebiete
# ===========================================================

tiles = []

lon = WEST

while lon < EAST:

    lon2 = min(lon + STEP, EAST)

    lat = SOUTH

    while lat < NORTH:

        lat2 = min(lat + STEP, NORTH)

        name = f"{lon:.1f}_{lat:.1f}.tif".replace("-", "m")

        outfile = TEMP / name

        if not outfile.exists():

            download_tile(
                lon,
                lon2,
                lat,
                lat2,
                outfile
            )

        tiles.append(outfile)

        lat = lat2

    lon = lon2

print()
print("Alle Kacheln heruntergeladen.")


# ===========================================================
# Mosaik
# ===========================================================

print("Erzeuge Mosaik...")

srcs = [rasterio.open(f) for f in tiles]

mosaic, transform = merge(srcs)

profile = srcs[0].profile.copy()

profile.update(
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform,
    compress="LZW",
    tiled=True,
)

with rasterio.open(MERGED, "w", **profile) as dst:
    dst.write(mosaic)

for s in srcs:
    s.close()

print("Mosaik erstellt.")


# ===========================================================
# Reprojection + 500 m
# ===========================================================

print("Reprojektion nach EPSG:3035")

with rasterio.open(MERGED) as src:

    transform, width, height = calculate_default_transform(
        src.crs,
        TARGET_CRS,
        src.width,
        src.height,
        *src.bounds,
        resolution=TARGET_RESOLUTION
    )

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        crs=TARGET_CRS,
        transform=transform,
        width=width,
        height=height,
        compress="LZW",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(OUTPUT, "w", **profile) as dst:

        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.average,
        )

print()
print("====================================")
print("FERTIG")
print("====================================")
print(OUTPUT)