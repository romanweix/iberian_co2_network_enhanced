#!/usr/bin/env python3

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

INPUT_FILE = "merged_srtm.tif"
OUTPUT_FILE = "merged_srtm_3035.tif"

TARGET_CRS = "EPSG:3035"

with rasterio.open(INPUT_FILE) as src:

    print("Eingabe:")
    print("CRS:", src.crs)
    print("Auflösung:", src.res)

    transform, width, height = calculate_default_transform(
        src.crs,
        TARGET_CRS,
        src.width,
        src.height,
        *src.bounds
    )

    profile = src.profile.copy()

    profile.update(
        crs=TARGET_CRS,
        transform=transform,
        width=width,
        height=height,
        compress="LZW",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:

        for band in range(1, src.count):

            reproject(
                source=rasterio.band(src, band),
                destination=rasterio.band(dst, band),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
            )

print(f"\nFertig: {OUTPUT_FILE}")