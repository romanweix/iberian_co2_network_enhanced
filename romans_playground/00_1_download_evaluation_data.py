#!/usr/bin/env python3

from pathlib import Path
import requests
import rasterio
import os
import time
import math


# -------------------------------------------------------
# Einstellungen
# -------------------------------------------------------

API_KEY = "288219a47ae7e0dfd26f824a81723a0d"

TMP_DIR = Path("tmp_dem")

URL = "https://portal.opentopography.org/API/globaldem"

MAX_RETRIES = 3


# gewünschte Gebiete:
# west, south, east, north

AREAS = {

    "SPAIN_PORTUGAL": (
        -10.0,
        36.0,
        4.5,
        44.0
    ),

    "PORTUGAL": (
        -11.0,
        36.0,
        -6.0,
        42.3
    ),

    "BALEARIC_ISLANDS": (
        0.5,
        38.0,
        4.5,
        40.2
    )
}


# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------

def create_filename(west, south):
    """
    Beispiel:
    west=-10.0 south=36.0

    -> m10.0_36.0.tif
    """

    if west < 0:
        west_str = f"m{abs(west):.1f}"
    else:
        west_str = f"{west:.1f}"

    return f"{west_str}_{south:.1f}.tif"



def generate_tiles(area):
    """
    Erzeugt 2° x 2° DEM-Kacheln,
    die eine Bounding Box vollständig abdecken.
    """

    west, south, east, north = area

    tiles = []

    start_lon = math.floor(west / 2) * 2
    end_lon = math.ceil(east / 2) * 2

    start_lat = math.floor(south / 2) * 2
    end_lat = math.ceil(north / 2) * 2


    lon = start_lon

    while lon < end_lon:

        lat = start_lat

        while lat < end_lat:

            tiles.append(
                create_filename(
                    lon,
                    lat
                )
            )

            lat += 2

        lon += 2


    return tiles



def filename_to_bbox(filename):

    stem = Path(filename).stem

    west_str, south_str = stem.split("_")

    west = float(
        west_str.replace("m", "-")
    )

    south = float(south_str)

    east = west + 2.0
    north = south + 2.0

    return west, east, south, north



def is_valid_geotiff(file):

    try:

        with rasterio.open(file) as ds:

            if ds.count < 1:
                return False

            ds.read(1)

        return True


    except Exception:

        print(
            f"Ungültige Datei: {file.name}"
        )

        return False



# -------------------------------------------------------
# Download
# -------------------------------------------------------

def download(file):

    west, east, south, north = filename_to_bbox(
        file.name
    )


    params = {

        "demtype": "COP30",

        "south": south,
        "north": north,
        "west": west,
        "east": east,

        "outputFormat": "GTiff",

        "API_Key": API_KEY
    }


    for attempt in range(MAX_RETRIES):

        print(
            f"{file.name}: Download Versuch {attempt+1}"
        )


        try:

            r = requests.get(
                URL,
                params=params,
                stream=True,
                timeout=120
            )


            if r.status_code != 200:

                print(r.text)

                time.sleep(3)

                continue



            with open(file, "wb") as f:

                for chunk in r.iter_content(
                    chunk_size=1024*1024
                ):

                    if chunk:
                        f.write(chunk)



            if is_valid_geotiff(file):

                print(
                    f"{file.name}: OK\n"
                )

                return


            else:

                os.remove(file)



        except Exception as e:

            print(e)

            if file.exists():
                os.remove(file)

            time.sleep(3)



    raise RuntimeError(
        f"{file.name} konnte nicht geladen werden"
    )



# -------------------------------------------------------
# Hauptprogramm
# -------------------------------------------------------

TMP_DIR.mkdir(
    exist_ok=True
)


# benötigte Dateien sammeln

required_files = set()


for name, area in AREAS.items():

    print(
        f"\nGebiet: {name}"
    )

    tiles = generate_tiles(area)

    print(
        f"{len(tiles)} Kacheln benötigt"
    )

    required_files.update(tiles)



print(
    f"\nGesamt benötigte Dateien: {len(required_files)}\n"
)



# -------------------------------------------------------
# prüfen und laden
# -------------------------------------------------------

for filename in sorted(required_files):

    filepath = TMP_DIR / filename


    if filepath.exists():

        if is_valid_geotiff(filepath):

            print(
                f"{filename}: vorhanden -> übersprungen"
            )

            continue


        else:

            print(
                f"{filename}: defekt -> löschen"
            )

            filepath.unlink()



    download(filepath)



print(
    "\nAlle DEM-Daten sind vollständig vorhanden."
)