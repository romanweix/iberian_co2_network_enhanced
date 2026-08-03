#!/usr/bin/env python3
"""
Merge aller Copernicus-Kacheln nach einem
500-m-DEM in EPSG:3035.

Ausgabe:
    merged_srtm.tif
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

INPUT_FOLDER = Path("tmp_dem")
OUTPUT_FILE = "merged_srtm_new1.tif"

TARGET_CRS = "EPSG:3035"
TARGET_RES = 500  # Meter


# ----------------------------------------------------------
# Dateien öffnen
# ----------------------------------------------------------

files = sorted(INPUT_FOLDER.glob("*.tif"))

datasets = []

for f in files:

    try:

        ds = rasterio.open(f)

        ds.read(1)

        datasets.append(ds)

        print("OK ", f.name)

    except Exception as e:

        print("Übersprungen:", f.name)
        print(e)


if len(datasets) == 0:
    raise RuntimeError("Keine gültigen Dateien gefunden.")


# ----------------------------------------------------------
# Mosaik
# ----------------------------------------------------------

print("Merge...")

mosaic, src_transform = merge(
    datasets,
    method="first"
)

src_crs = datasets[0].crs

for ds in datasets:
    ds.close()


# ----------------------------------------------------------
# Zielraster bestimmen
# ----------------------------------------------------------

left, bottom, right, top = rasterio.transform.array_bounds(
    mosaic.shape[1],
    mosaic.shape[2],
    src_transform
)

dst_transform, width, height = calculate_default_transform(
    src_crs,
    TARGET_CRS,
    mosaic.shape[2],
    mosaic.shape[1],
    left,
    bottom,
    right,
    top,
    resolution=TARGET_RES,
)

dst = np.empty(
    (1, height, width),
    dtype=np.float32
)

dst[:] = np.nan


# ----------------------------------------------------------
# Reprojection
# ----------------------------------------------------------

print("Reprojiziere auf 500 m...")

reproject(
    source=mosaic,
    destination=dst,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=dst_transform,
    dst_crs=TARGET_CRS,
    resampling=Resampling.average,
)


# ----------------------------------------------------------
# Schreiben
# ----------------------------------------------------------

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": TARGET_CRS,
    "transform": dst_transform,
    "compress": "LZW",
    "tiled": True,
    "BIGTIFF": "IF_SAFER",
    "nodata": np.nan,
}

print("Schreibe Datei...")

with rasterio.open(OUTPUT_FILE, "w", **profile) as out:

    out.write(dst)

print()
print("================================")
print("Fertig")
print("Datei:", OUTPUT_FILE)
print("Größe:", width, "x", height)
print("Auflösung:", TARGET_RES, "m")
print("CRS:", TARGET_CRS)
print("================================")