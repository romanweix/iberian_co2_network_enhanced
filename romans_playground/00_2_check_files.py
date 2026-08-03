from pathlib import Path
import rasterio

for f in Path("tmp_dem").glob("*.tif"):
    try:
        with rasterio.open(f) as ds:
            ds.read(1)
        print("OK ", f.name)
    except Exception as e:
        print("FEHLER:", f.name)
        print(e)