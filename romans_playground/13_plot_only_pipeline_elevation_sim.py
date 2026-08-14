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
SHEETNAME = "Pipeline candidates"
DEM = "merged_srtm_roman.tif"

STEP = 500  # Meter

ACTIVATE_PLOT = True

PIPE_ID = "10_a"

cricital_points = []


# ==========================================================
# DEM öffnen
# ==========================================================

src = rasterio.open(DEM)

#print(src.crs)

# Falls DEM in EPSG:3035 vorliegt
transformer = Transformer.from_crs(
    "EPSG:4326",
    src.crs,
    always_xy=True,
)

# ==========================================================
# Excel einlesen
# ==========================================================

df = pd.read_excel(EXCEL, sheet_name=SHEETNAME)

# Nur Onshore-Pipelines
df = df[df["Transport method"] == "Onshore"].reset_index(drop=True)

total_rows = len(df)

for idx, row in df.iterrows():
    geometry = row["geometry"]
    pipe_id = row["Pipeline identifier"]
    longitude = row["Longitude (km)"]

    if pipe_id == PIPE_ID:

        print("=" * 50)
        print(f"Zeile: {idx}/{total_rows}")
        print(f"Pipeline ID: {pipe_id}")
        print(f"Geometry: {geometry}")
        print(f"Longitude: {longitude}")
        print("-" * 50)

        line = wkt.loads(geometry)

        #print(line)
        #print(line.length)

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

        #print(f"Länge = {length/1000:.2f} km")

        import xarray as xr

        # ==========================================================
        # Temperaturkarte laden
        # ==========================================================

        temp_ds = xr.open_dataset("era5_january_mean_last5years.nc",
                                    engine="netcdf4")

        # Variable automatisch finden
        temp_var = list(temp_ds.data_vars)[0]
        temperature = temp_ds[temp_var]

        #print("Temperaturvariable:", temp_var)

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


        # ==========================================================
        # Erstelle DataFrame
        # ==========================================================
        profile = pd.DataFrame({
            "distance_m": distances,
            "distance_km": distances / 1000,
            "elevation_m": elevations,
            "temperature_degC": temperatures
        })

        #print(profile.head())

        profile.to_csv(
            "pipeline_profile.csv",
            index=False
        )

        import numpy as np
        import matplotlib.pyplot as plt
        import CoolProp.CoolProp as CP

        #print(CP.get_global_param_string("version"))

        activate_temp = True
        rho_dynamic = True

        # 1. Konstanten und Parameter
        step = 500          # Schrittweite in m
        x = np.arange(0, line_3035.length + step, step)

        # Durchmesser und Massenstrom (Mt/Jahr)
        diameters_inch = np.array([6, 10, 14, 18, 22, 26, 30, 34, 38, 42]) 
        diameters_m = diameters_inch * 0.0254

        diameters_m_paper = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        m_dot_mt_year_paper = np.array([1.7, 3.8, 6.7, 10.5, 15.2, 20.6, 26.9, 34.1, 42.1])

        # Kreisflächen [m²]
        area_paper = np.pi * (diameters_m_paper / 2)**2

        # Massenfluss pro Fläche [Mt/(a·m²)]
        m_dot_mt_year_area_paper = m_dot_mt_year_paper / area_paper

        mean_m_dot_mt_year_area_paper = np.mean(m_dot_mt_year_area_paper)

        m_dot_mt_year = mean_m_dot_mt_year_area_paper * (diameters_m / 2)**2 * np.pi

        print(diameters_m)
        print(m_dot_mt_year)
        
        mdot_reduction_pcnt = 0.6
        m_dot_kg_s = mdot_reduction_pcnt * m_dot_mt_year * 1e9 / (365 * 24 * 3600)
        #[ 32.34398782, 72.29832572, 127.47336377, 199.7716895, 
        # 289.19330289, 391.93302892, 511.79604262, 648.78234399, 800.98934551]

        # 2. Profile (Höhe und Außentemperatur)
        #h = 150 * np.sin(2 * np.pi * x / 100000) - 0.0006 * x + 250 
        h = elevations
        #t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [2, -2, 5, 5, 8])
        #t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [22, 18, 25, 15, 28])
        #t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [12,  8, 15,  5, 18])
        t_ext_profile = temperatures

        # 3. Stoffwerte & Startbedingungen
        p_start = 150e5     # 150 bar
        t_start = 25        # 25 °C
        u_values = [0.43, 2.0] 

        epsilon = 4.5e-5  # Roughness of the pipe in meters (for carbon steel)

        # statisches Dichte-Modell Pau
        temp_static = 298.15 # Temperature in Kelvin (25 °C)
        pressure_static = 125e5  # Pressure in Pa (125 bar)    

        # 4. Simulation
        if activate_temp:
            subplots_nr = 3
            height_ratios = [2, 2, 1]
        else:
            subplots_nr = 2
            height_ratios = [2, 1]
        fig, axes = plt.subplots(subplots_nr, len(u_values), figsize=(18, 14), gridspec_kw={'height_ratios': height_ratios}, sharex=True)
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(diameters_m)))

        # Konsolen-Header formatieren (dp = Druckänderung, dT = Temperaturänderung)
        header = f"{'U-Wert':<7} | {'D (inch)':<8} | {'x_end':<5} | {'x_end p':<7} | {'x_end t':<7} | {'x_pmin':<7} | {'p_min p':<7} | {'p_min t':<8} | {'x_tmin':<6} | {'t_min p':<7} | {'t_min t':<7}"
        print(header)
        print("-" * len(header))

        for col, u_val in enumerate(u_values):
            for i, (d_m, d_inch, m_dot) in enumerate(zip(diameters_m, diameters_inch, m_dot_kg_s)):
                p = np.zeros_like(x)
                t_in = np.zeros_like(x).astype(np.float64)
                p[0] = p_start
                t_in[0] = t_start
                
                area = np.pi * (d_m**2) / 4
                
                for j in range(len(x) - 1):
                    if activate_temp and rho_dynamic:
                        T_kelvin = t_in[j] + 273.15
                        P_pascal = p[j]

                        if T_kelvin < (32 + 273.15):
                            T_kelvin = 32 + 273.15
                        
                        if P_pascal < 100e5:
                            P_pascal = 100e5
                            
                        rho_curr = CP.PropsSI('D', 'P', P_pascal, 'T', T_kelvin, 'CO2')
                        cp_curr = CP.PropsSI('C', 'P', P_pascal, 'T', T_kelvin, 'CO2')
                        mu_jt_curr = CP.PropsSI('d(T)/d(P)|Hmass', 'P', P_pascal, 'T', T_kelvin, 'CO2')
                        mu_dynamic = CP.PropsSI('V', 'P', P_pascal, 'T', T_kelvin, 'CO2')
                    else:
                        rho_curr = CP.PropsSI('D', 'P', pressure_static, 'T', temp_static, 'CO2')
                        cp_curr = CP.PropsSI('C', 'P', pressure_static, 'T', temp_static, 'CO2')
                        mu_jt_curr = CP.PropsSI('d(T)/d(P)|Hmass', 'P', pressure_static, 'T', temp_static, 'CO2')
                        mu_dynamic = CP.PropsSI('V', 'P', pressure_static, 'T', temp_static, 'CO2')
                    
                    v =  m_dot / (rho_curr * area)
                    
                    re = (rho_curr * v * d_m) / mu_dynamic
                    f = (2 * np.log10(epsilon/3.7/d_m + 5.74/re**0.9))**-2 
                    dp_friction = f * (step / d_m) * (rho_curr * v**2 / 2)
                    dp_gravity = rho_curr * 9.81 * (h[j+1] - h[j])
                    
                    total_dp = dp_friction + dp_gravity
                    p[j+1] = p[j] - total_dp
                    
                    dq = u_val * (np.pi * d_m * step) * (t_ext_profile[j] - t_in[j])
                    dt_ambient = dq / (m_dot * cp_curr)
                    dt_jt = mu_jt_curr * (-total_dp) 
                    
                    if activate_temp:
                        t_in[j+1] = t_in[j] + dt_ambient + dt_jt

                # --- BERECHNUNG DER DIFFERENZEN (Startwert - Aktueller Wert) ---

                # 1. Ende der Leitung
                x_end_km = x[-1] / 1000
                dp_ende_bar = (p_start - p[-1]) / 1e5
                dT_ende_c = t_start - t_in[-1]

                # 2. Punkt mit dem höchsten Druckabfall (Minimaldruck)
                idx_p_min = np.argmin(p)

                # Falls der Minimaldruck am Ende liegt, den zweitkleinsten Druck verwenden
                if idx_p_min == len(p) - 1:
                    idx_p_min = np.argsort(p)[1]

                x_dpmax_km = x[idx_p_min] / 1000
                dp_max_bar = (p_start - p[idx_p_min]) / 1e5
                dT_dpmax_c = t_start - t_in[idx_p_min]

                # 3. Kältester Punkt (Minimum der Temperatur)
                idx_t_min = np.argmin(t_in)

                # Falls die Minimaltemperatur am Ende liegt, die zweitniedrigste Temperatur verwenden
                if (idx_t_min == len(t_in) - 1):
                    idx_t_min = np.argsort(t_in)[1]
                    if (idx_t_min == idx_p_min):
                        idx_t_min = np.argsort(t_in)[2]
                elif (idx_t_min == idx_p_min):
                    idx_t_min = np.argsort(t_in)[1]

                x_tmin_km = x[idx_t_min] / 1000
                dp_tmin_bar = (p_start - p[idx_t_min]) / 1e5
                dT_tmin_c = t_start - t_in[idx_t_min]

                cricital_points.append({
                    "Pipe ID": pipe_id,
                    "Longitude [km]": longitude,
                    "U-Wert [W/m^2/K]": u_val,
                    "D [inch]": d_inch,
                    "p_end [bar]": (p[0] / 1e5) - dp_ende_bar,
                    "t_end [°C]": t_in[0] - dT_ende_c,
                    "p_min p[bar]": p[idx_p_min] / 1e5,
                    "p_min T [°C]": t_in[idx_p_min],
                    "t_min p [bar]": p[idx_t_min] / 1e5,
                    "t_min T[°C]": t_in[idx_t_min],
                    "geometry": geometry
                })

                # Ausgabe in der Konsole
                print(f"{u_val:<7} | {d_inch:<8.1f} | {x_end_km:<5.0f} | {(p[0] / 1e5) - dp_ende_bar:<7.1f} | {t_in[0] - dT_ende_c:<7.1f} | {x_dpmax_km:<7.1f} | {p[idx_p_min] / 1e5:<7.1f} | {t_in[idx_p_min]:<8.1f} | {x_tmin_km:<6.1f} | {p[idx_t_min] / 1e5:<7.1f} | { t_in[idx_t_min]:<7.1f}")

                # Plots (bleiben absolut)
                axes[0, col].plot(x/1000, p/1e5, color=colors[i], label=f"D={d_inch}inch")

                axes[0, col].scatter(
                    x[idx_p_min]/1000,
                    p[idx_p_min]/1e5,
                    color=colors[i],
                    edgecolor='black',
                    s=80,
                    marker='o',
                    zorder=5
                )

                if activate_temp:
                    axes[1, col].plot(x/1000, t_in, color=colors[i])
                    
                    axes[1, col].scatter(
                        x[idx_t_min]/1000,
                        t_in[idx_t_min],
                        color=colors[i],
                        edgecolor='black',
                        s=80,
                        marker='o',
                        zorder=5
                    )

            if ACTIVATE_PLOT:   
                # --- Beschriftung der Spalten ---
                axes[0, col].set_title(f"\ninsulation U = {u_val} W/m²K", fontsize=14, fontweight='bold')

                # --- Subplot 1: Druck ---
                axes[0, col].axhline(73.8, color='red', linestyle='--', alpha=0.5, label="supercrit threshold = 73.8 bar")
                axes[0, col].grid(True, alpha=0.2)
                if col == 0: axes[0, col].set_ylabel("\npressure [bar]")
                axes[0, col].legend(fontsize='x-small')

                # --- Subplot 2: Temperatur ---
                if activate_temp:
                    axes[1, col].plot(x/1000, t_ext_profile, 'k--', alpha=0.4, label="ambient temp")
                    axes[1, col].axhline(31.1, color='red', linestyle='--', alpha=0.5, label="supercrit threshold = 31.1°C")
                    #axes[1, col].axhline(10.0, color='red', linestyle='-', alpha=0.5, label="ice threshold = 10.0°C")
                    axes[1, col].grid(True, alpha=0.2)
                    if col == 0: axes[1, col].set_ylabel("\ntemperature [°C]")
                    axes[1, col].legend(fontsize='x-small')
                    height_subplot_row = 2
                else:
                    height_subplot_row = 1
                # --- Subplot 3: Höhe ---
                axes[height_subplot_row, col].fill_between(x/1000, h, color='brown', alpha=0.2)
                axes[height_subplot_row, col].grid(True, alpha=0.2)
                if col == 0: axes[height_subplot_row, col].set_ylabel("\ntopology [m]")
                axes[height_subplot_row, col].set_xlabel("distance [km]\n\n")

        if ACTIVATE_PLOT:
            plt.tight_layout()
            plt.show()

src.close()

"""
import pandas as pd

file_path = "sim_pipeline_candidates_45degC.xlsx"

# deine berechneten Ergebnisse
df_results = pd.DataFrame(cricital_points)

with pd.ExcelWriter(
    file_path,
    engine="openpyxl",
    mode="a",              # bestehende Datei öffnen
    if_sheet_exists="replace"  # vorhandenen Reiter überschreiben
) as writer:
    df_results.to_excel(
        writer,
        sheet_name="Pipeline sim. results",
        index=False
    )

"""