import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

#print(CP.get_global_param_string("version"))

activate_temp = False
rho_dynamic = False

# 1. Konstanten und Parameter
L = 150000          # Gesamtlänge in m
step = 500          # Schrittweite in m
x = np.arange(0, L + step, step)

# Durchmesser (m) und Massenstrom (Mt/Jahr)
diameters = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) 
m_dot_mt_year = np.array([1.7, 3.8, 6.7, 10.5, 15.2, 20.6, 26.9, 34.1, 42.1])
mdot_reduction_pcnt = 0.6
m_dot_kg_s = mdot_reduction_pcnt * m_dot_mt_year * 1e9 / (365 * 24 * 3600)
#[ 32.34398782, 72.29832572, 127.47336377, 199.7716895, 
# 289.19330289, 391.93302892, 511.79604262, 648.78234399, 800.98934551]

# 2. Profile (Höhe und Außentemperatur)
h = 150 * np.sin(2 * np.pi * x / 100000) - 0.0006 * x + 250 
#t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [2, -2, 5, 5, 8])
#t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [22, 18, 25, 15, 28])
t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [12,  8, 15,  5, 18])

# 3. Stoffwerte & Startbedingungen
p_start = 150e5     # 150 bar
t_start = 40        # 40 °C
u_values = [0.43, 2.0] 

epsilon = 4.5e-5  # Roughness of the pipe in meters (for carbon steel)

# statisches Dichte-Modell Pau
temp_static = 308.15  # Temperature in Kelvin (35 °C)
pressure_static = 125e5  # Pressure in Pa (125 bar)    

# 4. Simulation
if activate_temp:
    subplots_nr = 3
    height_ratios = [2, 2, 1]
else:
    subplots_nr = 2
    height_ratios = [2, 1]
fig, axes = plt.subplots(subplots_nr, len(u_values), figsize=(18, 14), gridspec_kw={'height_ratios': height_ratios}, sharex=True)
colors = plt.cm.plasma(np.linspace(0, 0.8, len(diameters)))

# Konsolen-Header formatieren (dp = Druckänderung, dT = Temperaturänderung)
header = f"{'U-Wert':<7} | {'D (m)':<5} | {'x_End':<5} | {'dp_End':<7} | {'dT_End':<7} | {'x_dpmax':<7} | {'dp_max':<7} | {'dT_dpmax':<8} | {'x_Tmin':<6} | {'dp_Tmin':<7} | {'dT_Tmin':<7}"
print(header)
print("-" * len(header))

for col, u_val in enumerate(u_values):
    for i, (d, m_dot) in enumerate(zip(diameters, m_dot_kg_s)):
        p = np.zeros_like(x)
        t_in = np.zeros_like(x).astype(np.float64)
        p[0] = p_start
        t_in[0] = t_start
        
        area = np.pi * (d**2) / 4
        
        for j in range(len(x) - 1):
            if activate_temp and rho_dynamic:
                T_kelvin = t_in[j] + 273.15
                P_pascal = p[j]

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
            
            re = (rho_curr * v * d) / mu_dynamic
            f = (2 * np.log10(epsilon/3.7/d + 5.74/re**0.9))**-2 
            dp_friction = f * (step / d) * (rho_curr * v**2 / 2)
            dp_gravity = rho_curr * 9.81 * (h[j+1] - h[j])
            
            total_dp = dp_friction + dp_gravity
            p[j+1] = p[j] - total_dp
            
            dq = u_val * (np.pi * d * step) * (t_ext_profile[j] - t_in[j])
            dt_ambient = dq / (m_dot * cp_curr)
            dt_jt = mu_jt_curr * (-total_dp) 
            
            if activate_temp:
                t_in[j+1] = t_in[j] + dt_ambient + dt_jt

        # --- BERECHNUNG DER DIFFERENZEN (Startwert - Aktueller Wert) ---
        # 1. Ende der Leitung (x = 150 km)
        x_end_km = x[-1] / 1000
        dp_ende_bar = (p_start - p[-1]) / 1e5
        dT_ende_c = t_start - t_in[-1]
        
        # 2. GEÄNDERT: Punkt mit dem HÖCHSTEN Druckabfall finden (entspricht minimalem Absolutdruck)
        idx_p_min = np.argmin(p) 
        x_dpmax_km = x[idx_p_min] / 1000
        dp_max_bar = (p_start - p[idx_p_min]) / 1e5
        dT_dpmax_c = t_start - t_in[idx_p_min]
        
        # 3. Kältester Punkt (Minimum der Temperaturkurve)
        idx_t_min = np.argmin(t_in)
        x_tmin_km = x[idx_t_min] / 1000
        dp_tmin_bar = (p_start - p[idx_t_min]) / 1e5
        dT_tmin_c = t_start - t_in[idx_t_min]

        # Ausgabe in der Konsole
        print(f"{u_val:<7} | {d:<5.1f} | {x_end_km:<5.0f} | {dp_ende_bar:<7.1f} | {dT_ende_c:<7.1f} | {x_dpmax_km:<7.1f} | {dp_max_bar:<7.1f} | {dT_dpmax_c:<8.1f} | {x_tmin_km:<6.1f} | {dp_tmin_bar:<7.1f} | {dT_tmin_c:<7.1f}")

        # Plots (bleiben absolut)
        axes[0, col].plot(x/1000, p/1e5, color=colors[i], label=f"D={d}m")
        if activate_temp:
            axes[1, col].plot(x/1000, t_in, color=colors[i])
    
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

plt.tight_layout()
plt.show()