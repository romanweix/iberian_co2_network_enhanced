import rasterio
from rasterio.warp import transform_bounds


DEM_FILE = "data/merged_srtm.tif"


# gewünschte Bereiche
BBOXES = {
    "SPAIN_PORTUGAL": (-10.0, 36.0, 4.5, 44.0),
    "PORTUGAL": (-11.0, 36.0, -6.0, 42.3),
    "BALEARIC_ISLANDS": (0.5, 38.0, 4.5, 40.2),
}


def bbox_inside(dem_bounds, test_bbox):
    """
    Prüft, ob test_bbox vollständig im DEM liegt.

    Reihenfolge:
    lon_min, lat_min, lon_max, lat_max
    """

    lon_min, lat_min, lon_max, lat_max = test_bbox

    return (
        lon_min >= dem_bounds.left
        and lon_max <= dem_bounds.right
        and lat_min >= dem_bounds.bottom
        and lat_max <= dem_bounds.top
    )


with rasterio.open(DEM_FILE) as src:

    print("DEM CRS:")
    print(src.crs)

    print("\nOriginal Bounds:")
    print(src.bounds)

    # nach WGS84 transformieren
    dem_wgs84 = transform_bounds(
        src.crs,
        "EPSG:4326",
        *src.bounds
    )

    dem_bbox = rasterio.coords.BoundingBox(
        left=dem_wgs84[0],
        bottom=dem_wgs84[1],
        right=dem_wgs84[2],
        top=dem_wgs84[3],
    )

    print("\nDEM Abdeckung WGS84:")
    print(
        f"lon: {dem_bbox.left:.3f} bis {dem_bbox.right:.3f}"
    )
    print(
        f"lat: {dem_bbox.bottom:.3f} bis {dem_bbox.top:.3f}"
    )

    print("\nPrüfung:")
    
    for name, bbox in BBOXES.items():

        covered = bbox_inside(
            dem_bbox,
            bbox
        )

        if covered:
            print(f"✓ {name}: vollständig abgedeckt")

        else:
            print(f"✗ {name}: NICHT vollständig abgedeckt")