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

SHOW_CRITICAL_POINTS = False  # Marker für lokales Druck-/Temperaturminimum

if SHOW_CRITICAL_POINTS == True:
    TEXT_CRIT_POINTS = "critPnts"
else:
    TEXT_CRIT_POINTS = "nocritPnts"

PIPE_ID = "5_a"

cricital_points = []

activate_temp = True
rho_dynamic = False
DENSE_ACTIVE = False

if activate_temp == True:
    TEXT_ACT_TEMP = "temp"
else:
    TEXT_ACT_TEMP = "notemp"

if rho_dynamic == True:
    TEXT_DYN = "dyn"
else:
    TEXT_DYN = "notdyn"


if DENSE_ACTIVE == True:
    MIN_TEMP_C = 15
    STARTING_TEMP_C = 25
    STATIC_AVG_TEMP_C = 25
    TEXT_TEMP_LIMIT = "$T_\mathrm{ice}$(CO$_2$) = 15.0 $^{\circ}$C"
    GRAPH_TEMP_MIN_C = 15.0
    TEXT_PHASE = "dense"
else: # supercricital co2
    MIN_TEMP_C = 32
    STARTING_TEMP_C = 45
    STATIC_AVG_TEMP_C = 35
    TEXT_TEMP_LIMIT = "$T_\mathrm{crit}$(CO$_2$) = 31.1 $^{\circ}$C"
    GRAPH_TEMP_MIN_C = 31.1
    TEXT_PHASE = "critical"





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
        t_start = STARTING_TEMP_C        # 25 °C
        u_values = [0.43, 2.0] 

        epsilon = 4.5e-5  # Roughness of the pipe in meters (for carbon steel)

        # statisches Dichte-Modell Pau
        temp_static = 273.15 + STATIC_AVG_TEMP_C # Temperature in Kelvin (25 °C)
        pressure_static = 125e5  # Pressure in Pa (125 bar)    

        # 4. Simulation: Druck- und Temperaturprofile berechnen
        # Konsolen-Header formatieren (dp = Druckänderung, dT = Temperaturänderung)
        header = f"{'U-Wert':<7} | {'D (inch)':<8} | {'x_end':<5} | {'x_end p':<7} | {'x_end t':<7} | {'x_pmin':<7} | {'p_min p':<7} | {'p_min t':<8} | {'x_tmin':<6} | {'t_min p':<7} | {'t_min t':<7}"
        print(header)
        print("-" * len(header))

        # results[u_val][d_inch] = {"p": ..., "t": ..., "idx_p_min": ..., "idx_t_min": ...}
        results = {}

        for u_val in u_values:
            results[u_val] = {}
            for d_m, d_inch, m_dot in zip(diameters_m, diameters_inch, m_dot_kg_s):
                p = np.zeros_like(x)
                t_in = np.zeros_like(x).astype(np.float64)
                p[0] = p_start
                t_in[0] = t_start
                
                area = np.pi * (d_m**2) / 4
                
                for j in range(len(x) - 1):
                    if activate_temp and rho_dynamic:
                        T_kelvin = t_in[j] + 273.15
                        P_pascal = p[j]

                        if T_kelvin < (MIN_TEMP_C + 273.15):
                            T_kelvin = MIN_TEMP_C + 273.15
                        
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

                results[u_val][d_inch] = {
                    "p": p,
                    "t": t_in,
                    "idx_p_min": idx_p_min,
                    "idx_t_min": idx_t_min,
                }

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

        # ==========================================================
        # 5. Publikationsreife Darstellung
        # ==========================================================
        if ACTIVATE_PLOT:

            import os
            from matplotlib.colors import Normalize
            from matplotlib.transforms import Bbox

            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "font.size": 9,
                "axes.titlesize": 9.5,
                "axes.labelsize": 9.5,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 7.5,
                "axes.linewidth": 0.7,
                "lines.linewidth": 1.1,
                "grid.linewidth": 0.5,
                "grid.alpha": 0.35,
                "axes.grid": True,
                "axes.axisbelow": True,
                "figure.dpi": 130,
                "savefig.dpi": 300,
            })

            cmap = plt.cm.viridis
            norm = Normalize(vmin=diameters_inch.min(), vmax=diameters_inch.max())
            colors = cmap(norm(diameters_inch))
            x_km = x / 1000

            # Ohne Temperatur entfällt die dritte Zeile komplett (nicht nur
            # leer gelassen) - Figur, Gridspec und Colorbar-Höhe passen sich an.
            fig = plt.figure(figsize=(8.0, 7.6 if activate_temp else 5.55))
            if activate_temp:
                gs = fig.add_gridspec(
                    3, 4,
                    height_ratios=[0.85, 1.35, 1.35],
                    width_ratios=[1, 1, 0.30, 0.035],
                    hspace=0.22, wspace=0.06,
                    left=0.09, right=0.85, top=0.92, bottom=0.07,
                )
            else:
                gs = fig.add_gridspec(
                    2, 4,
                    height_ratios=[0.85, 1.35],
                    width_ratios=[1, 1, 0.30, 0.035],
                    hspace=0.22, wspace=0.06,
                    left=0.09, right=0.85, top=0.90, bottom=0.10,
                )

            ax_p = [fig.add_subplot(gs[1, 0])]
            ax_p.append(fig.add_subplot(gs[1, 1], sharey=ax_p[0]))
            ax_elev = fig.add_subplot(gs[0, 0:2], sharex=ax_p[0])

            if activate_temp:
                ax_t = [fig.add_subplot(gs[2, 0], sharex=ax_p[0])]
                ax_t.append(fig.add_subplot(gs[2, 1], sharex=ax_p[0], sharey=ax_t[0]))
                cax = fig.add_subplot(gs[1:, 3])
            else:
                ax_t = [None, None]
                cax = fig.add_subplot(gs[1, 3])

            # --- Höhenprofil: identisch für beide U-Werte, daher nur ein Panel ---
            ax_elev.fill_between(x_km, h, color="#8c6a4a", alpha=0.30, lw=0)
            ax_elev.plot(x_km, h, color="#6b4d30", lw=0.9)
            ax_elev.set_ylabel("elevation\n[m a.s.l.]")
            ax_elev.set_ylim(bottom=0)
            ax_elev.tick_params(labelbottom=False)

            marker_kw = dict(edgecolor="black", linewidth=0.6, s=24, marker="o", zorder=5)

            def add_elevation_backdrop(ax, show_axis):
                # Topologie transparent hinter den Daten einblenden; rechte y-Achse
                # dient als Einheitenachse. Skalierung so gewählt, dass das Relief
                # nur die untere Haelfte des Panels einnimmt und die Kurven nicht überdeckt.
                ax_e = ax.twinx()
                ax_e.fill_between(x_km, h, color="#8c6a4a", alpha=0.18, lw=0, zorder=0)
                ax_e.set_ylim(0, max(h) * 2.3)
                ax_e.grid(False)
                ax.set_zorder(ax_e.get_zorder() + 1)
                ax.patch.set_visible(False)
                if show_axis:
                    ax_e.set_ylabel("elevation [m a.s.l.]", color="#6b4d30")
                    ax_e.tick_params(axis="y", colors="#6b4d30", labelsize=7)
                else:
                    ax_e.set_yticks([])
                return ax_e

            for col, u_val in enumerate(u_values):
                axp, axt = ax_p[col], ax_t[col]

                add_elevation_backdrop(axp, show_axis=(col == 1))
                if activate_temp:
                    add_elevation_backdrop(axt, show_axis=(col == 1))

                for i, d_inch in enumerate(diameters_inch):
                    res = results[u_val][d_inch]
                    p_i, t_i = res["p"], res["t"]
                    idx_p_min, idx_t_min = res["idx_p_min"], res["idx_t_min"]

                    axp.plot(x_km, p_i / 1e5, color=colors[i])
                    if SHOW_CRITICAL_POINTS:
                        axp.scatter(x_km[idx_p_min], p_i[idx_p_min] / 1e5, color=colors[i], **marker_kw)

                    if activate_temp:
                        axt.plot(x_km, t_i, color=colors[i])
                        if SHOW_CRITICAL_POINTS:
                            axt.scatter(x_km[idx_t_min], t_i[idx_t_min], color=colors[i], **marker_kw)

                # --- Druck-Panel ---
                ylim = axp.get_ylim()
                axp.axhspan(ylim[0], 0, color="0.75", alpha=0.55, zorder=0)
                axp.set_ylim(ylim)
                axp.axhline(73.8, color="firebrick", ls="--", lw=1.0, alpha=0.85,
                            label=r"$p_\mathrm{crit}$(CO$_2$) = 73.8 bar")
                insulation_note = "well insulated" if u_val == min(u_values) else "poorly insulated"
                axp.set_title(
                    rf"$U$ = {u_val:g} W m$^{{-2}}$ K$^{{-1}}$ ({insulation_note})",
                    fontweight="bold",
                )

                # --- Temperatur-Panel ---
                if activate_temp:
                    axp.tick_params(labelbottom=False)
                    axt.plot(x_km, t_ext_profile, color="0.35", ls="--", lw=1.0, label="ambient temperature")
                    axt.axhline(GRAPH_TEMP_MIN_C, color="firebrick", ls="--", lw=1.0, alpha=0.85,
                                label=TEXT_TEMP_LIMIT)
                    axt.set_xlabel("distance along pipeline [km]")
                else:
                    # Druck-Panel ist ohne Temperatur-Zeile die unterste Reihe
                    axp.set_xlabel("distance along pipeline [km]")

            # --- Legenden (einmal je Zeile, linke Spalte) ---
            legend_kw = dict(frameon=True, facecolor="white", framealpha=0.85,
                              edgecolor="none", borderpad=0.4)

            h_p, l_p = ax_p[0].get_legend_handles_labels()
            if SHOW_CRITICAL_POINTS:
                proxy_kw = dict(marker="o", linestyle="", markerfacecolor="0.5",
                                 markeredgecolor="black", markeredgewidth=0.6, markersize=5)
                crit_p_proxy = plt.Line2D([], [], label="local pressure minimum", **proxy_kw)
                h_p, l_p = h_p + [crit_p_proxy], l_p + [crit_p_proxy.get_label()]
            ax_p[0].legend(h_p, l_p, loc="upper left", bbox_to_anchor=(0.01, 0.90), **legend_kw)

            if activate_temp:
                h_t, l_t = ax_t[0].get_legend_handles_labels()
                if SHOW_CRITICAL_POINTS:
                    crit_t_proxy = plt.Line2D([], [], label="local temperature minimum", **proxy_kw)
                    h_t, l_t = h_t + [crit_t_proxy], l_t + [crit_t_proxy.get_label()]
                ax_t[0].legend(h_t, l_t, loc="lower left", **legend_kw)

            ax_p[0].set_ylabel("pressure [bar]")
            tick_hide_axes = [ax_p[1]]
            panel_axes = [ax_elev, ax_p[0], ax_p[1]]
            if activate_temp:
                ax_t[0].set_ylabel("temperature [$^{\\circ}$C]")
                tick_hide_axes.append(ax_t[1])
                panel_axes += [ax_t[0], ax_t[1]]
            for axr in tick_hide_axes:
                axr.tick_params(labelleft=False)

            label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5)
            for ax, lbl in zip(panel_axes, "abcde"):
                ax.text(0.012, 0.94, f"({lbl})", transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="top", ha="left", bbox=label_bbox)

            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cbar.set_label("pipeline diameter $D$ [in]")
            cbar.set_ticks(diameters_inch)

            fig.suptitle(
                f"Pipeline {pipe_id} — steady-state pressure and temperature profile "
                f"($L$ = {x_km[-1]:.0f} km, flow reduced to {int(mdot_reduction_pcnt * 100)}% of design capacity)",
                fontsize=10.5, fontweight="bold",
            )

            # --- Export: Übersichtsabbildung + einzelne Panels ---
            outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
            os.makedirs(outdir, exist_ok=True)
            base = f"pipeline_{pipe_id}_profile_{TEXT_PHASE}_{TEXT_ACT_TEMP}_{TEXT_DYN}_{TEXT_CRIT_POINTS}"

            fig.savefig(os.path.join(outdir, f"{base}.png"), bbox_inches="tight")
            #fig.savefig(os.path.join(outdir, f"{base}.pdf"), bbox_inches="tight")

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            def save_group(axes_group, suffix, pad=0.04):
                # kleine, feste Randbreite statt relativer Skalierung, damit
                # benachbarte Panels (z.B. Elevation-Zeile) nicht mit angeschnitten werden
                bbox = Bbox.union(
                    [a.get_tightbbox(renderer) for a in axes_group]
                ).transformed(fig.dpi_scale_trans.inverted())
                bbox = bbox.padded(pad)
                fig.savefig(os.path.join(outdir, f"{base}_{suffix}.png"), bbox_inches=bbox)
                #fig.savefig(os.path.join(outdir, f"{base}_{suffix}.pdf"), bbox_inches=bbox)

            save_group([ax_elev], "elevation")
            save_group([ax_p[0], ax_p[1], cax], "pressure")
            if activate_temp:
                save_group([ax_t[0], ax_t[1], cax], "temperature")

            print(f"Figures written to: {outdir}")

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