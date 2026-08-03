import numpy as np
import matplotlib.pyplot as plt

activate_temp = True
rho_dynamic = True

# 1. Konstanten und Parameter
L = 150000          # Gesamtlänge in m
step = 500          # Schrittweite in m
x = np.arange(0, L + step, step)

# Durchmesser (m) und Massenstrom (Mt/Jahr)
diameters = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]) 
m_dot_mt_year = np.array([1.7, 3.8, 6.7, 10.5, 15.2, 20.6, 26.9, 42.1])
m_dot_kg_s = m_dot_mt_year * 1e9 / (365 * 24 * 3600)

# 2. Profile (Höhe und Außentemperatur)
# Ein markantes Höhenprofil für die Visualisierung
#h = x*0 
#h = 150 * np.sin(2 * np.pi * x / 100000) - 0.0006 * x + 250 
h = 150 * np.sin(2 * np.pi * x / 100000) - 0.0006 * x + 250 
# Außentemperatur-Verlauf (Luft)
t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [2, -2, 5, 5, 8])
#t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [12, 8, 15, 5, 18])
#t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [22, 18, 25, 15, 28])
#t_ext_profile = np.interp(x, [0, 40000, 80000, 120000, 150000], [2, 2, 2, 2, 2])

# 3. Stoffwerte & Startbedingungen
p_start = 150e5     # 150 bar
t_start = 40        # 40 °C
mu = 0.00006        # Dynamische Viskosität
cp = 2100           # Spezifische Wärmekapazität
mu_jt = 1.1e-6      # Joule-Thomson-Koeffizient (K/Pa)
#u_values = [0.3, 1.0, 5.0] # Realistischer U-Wert (W/m²K)
u_values = [0.3, 5.0] # Realistischer U-Wert (W/m²K)

# Dichte-Modell (linearisiert)
rho_ref = 800       
t_ref = 20          
d_rho_dt = -6.5     

# 4. Simulation
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
colors = plt.cm.plasma(np.linspace(0, 0.8, len(diameters)))

# Index des höchsten Punktes finden
idx_h_max = np.argmax(h)
dist_h_max = x[idx_h_max] / 1000

print(f"{'U-Wert':<10} | {'D (m)':<6} | {'dp @ max_h (bar)':<18} | {'dp @ Ende (bar)':<15}")
print("-" * 60)

for col, u_val in enumerate(u_values):
    for i, (d, m_dot) in enumerate(zip(diameters, m_dot_kg_s)):
        p = np.zeros_like(x)
        t_in = np.zeros_like(x).astype(np.float64)
        p[0] = p_start
        t_in[0] = t_start
        
        area = np.pi * (d**2) / 4
        
        for j in range(len(x) - 1):
            # Dichte berechnen
            if activate_temp and rho_dynamic:
                rho_curr = rho_ref + d_rho_dt * (t_in[j] - t_ref)
            else:
                rho_curr = rho_ref
            
            v = m_dot / (rho_curr * area)
            
            # Druckänderungen
            re = (rho_curr * v * d) / mu
            f = (1.8 * np.log10(re/6.9))**-2 
            dp_friction = f * (step / d) * (rho_curr * v**2 / 2)
            dp_gravity = rho_curr * 9.81 * (h[j+1] - h[j])
            
            total_dp = dp_friction + dp_gravity
            p[j+1] = p[j] - total_dp
            
            # Temperaturänderungen
            dq = u_val * (np.pi * d * step) * (t_ext_profile[j] - t_in[j])
            dt_ambient = dq / (m_dot * cp)
            dt_jt = mu_jt * (-total_dp) 
            
            if activate_temp:
                t_in[j+1] = t_in[j] + dt_ambient + dt_jt

        # --- BERECHNUNG DER DRUCKABFÄLLE ---
        # Druckabfall = Startdruck - aktueller Druck
        dp_at_h_max = (p_start - p[idx_h_max]) / 1e5
        dp_at_end = (p_start - p[-1]) / 1e5
        
        # Ausgabe in der Konsole
        print(f"{u_val:<10} | {d:<6.1f} | {dp_at_h_max:<18.2f} | {dp_at_end:<15.2f}")

        # Plots (bleiben gleich)
        axes[0, col].plot(x/1000, p/1e5, color=colors[i], label=f"D={d}m")
        if activate_temp:
            axes[1, col].plot(x/1000, t_in, color=colors[i])
    
    # --- Beschriftung der Spalten ---
    axes[0, col].set_title(f"Isolierung U = {u_val} W/m²K", fontsize=14, fontweight='bold')

    # --- Subplot 1: Druck ---
    axes[0, col].axhline(73.8, color='red', linestyle='--', alpha=0.5)
    axes[0, col].grid(True, alpha=0.2)
    if col == 0: axes[0, col].set_ylabel("Innendruck (bar)")
    axes[0, col].legend(fontsize='x-small')

    # --- Subplot 2: Temperatur ---
    if activate_temp:
        axes[1, col].plot(x/1000, t_ext_profile, 'k--', alpha=0.4, label="T_ext")
        axes[1, col].axhline(31.1, color='red', linestyle='--', alpha=0.5)
        axes[1, col].axhline(5.0, color='red', linestyle='-', alpha=0.5)
        axes[1, col].grid(True, alpha=0.2)
        if col == 0: axes[1, col].set_ylabel("Temperatur (°C)")

    # --- Subplot 3: Höhe ---
    axes[2, col].fill_between(x/1000, h, color='brown', alpha=0.2)
    axes[2, col].grid(True, alpha=0.2)
    if col == 0: axes[2, col].set_ylabel("Höhe (m)")
    axes[2, col].set_xlabel("Distanz (km)")

# # --- Subplot 1: Druck ---
# ax1.set_ylabel("Innendruck (bar)")
# ax1.set_title("Gekoppelte CO2-Pipeline Simulation (150 km)", fontsize=14)
# ax1.axhline(73.8, color='red', linestyle='--', alpha=0.7, label="Kritischer Druck (73.8 bar)")
# ax1.legend(loc='upper right', ncol=2, fontsize='small')
# ax1.grid(True, alpha=0.2)

# # --- Subplot 2: Temperaturen ---
# if activate_temp:
#     ax2.plot(x/1000, t_ext_profile, 'k--', linewidth=1.5, label="Außentemperatur (Luft)")
#     ax2.axhline(31.1, color='red', linestyle='--', alpha=0.7, label="Kritische Temperatur (31.1 °C)")
#     ax2.set_ylabel("Temperatur (°C)")
#     ax2.legend(loc='upper right')
#     ax2.grid(True, alpha=0.2)

# # --- Subplot 3: Höhenprofil ---
# ax3.fill_between(x/1000, h, color='brown', alpha=0.3, label="Geländeprofil")
# ax3.set_ylabel("Höhe über NN (m)")
# ax3.set_xlabel("Pipeline-Länge (km)")
# ax3.set_ylim(0, max(h) + 100)
# ax3.grid(True, alpha=0.2)
# ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()