import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# --- 1. Parameter ---
L = 150000          
m_dot = 150         
P_inlet = 125e5        
U_value = 2.5          
dx = 500            
n_steps = int(L / dx)
eta_pumpe = 0.75    
epsilon = 0.000045
max_boost = 50e5 

# Umgebungstemperatur (Wüstenszenario: 35°C bis 40°C)
T_amb_array = np.linspace(18 + 273.15, 18 + 273.15, n_steps + 1)
dist_full_km = np.linspace(0, L, n_steps + 1) / 1000

# Suchbereich für Starttemperaturen
test_temps_zone1 = np.linspace(16, 24, 5) 
test_temps_zone2 = np.linspace(36, 65, 5) 

diameters = [0.4, 0.6, 0.8] 
# Feste Farben für die Durchmesser definieren
color_list = ['#E63946', '#2A9D8F', '#457B9D']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- 2. Simulation mit Energie-Optimierung ---
for idx, D in enumerate(diameters):
    best_score = float('inf') 
    best_start_T = None
    best_zone_params = None
    current_color = color_list[idx % len(color_list)]

    # Vorab-Optimierung: Beste Starttemperatur finden
    for zone_idx, test_set in enumerate([test_temps_zone1, test_temps_zone2]):
        z_min, z_max = (15.0, 25.0) if zone_idx == 0 else (35.0, 1000.0)
        
        for T_start_test in test_set:
            total_energy = 0
            station_count = 0
            T_curr, P_curr = T_start_test + 273.15, P_inlet
            
            for i in range(n_steps):
                T_c = T_curr - 273.15
                if (P_curr < 100.5e5) or (T_c < z_min) or (T_c > z_max):
                    station_count += 1
                    try:
                        h_before = PropsSI('H', 'T', T_curr, 'P', P_curr, 'CO2')
                        P_after = P_curr + max_boost
                        s_in = PropsSI('S', 'T', T_curr, 'P', P_curr, 'CO2')
                        h_is = PropsSI('H', 'P', P_after, 'S', s_in, 'CO2')
                        h_pumped = h_before + (h_is - h_before) / eta_pumpe
                        h_final = PropsSI('H', 'T', T_start_test + 273.15, 'P', P_after, 'CO2')
                        total_energy += abs(h_pumped - h_before) + abs(h_final - h_pumped)
                        T_curr, P_curr = T_start_test + 273.15, P_after
                    except: break
                
                try:
                    rho = PropsSI('D', 'T', T_curr, 'P', P_curr, 'CO2')
                    mu = PropsSI('V', 'T', T_curr, 'P', P_curr, 'CO2')
                    v = m_dot / (rho * (np.pi * (D/2)**2))
                    Re = (rho * v * D) / mu
                    f = 0.25 / (np.log10(epsilon/(3.7*D) + 5.74/(Re**0.9)))**2
                    P_curr += -f * (rho * v**2) / (2 * D) * dx
                    cp = PropsSI('C', 'T', T_curr, 'P', P_curr, 'CO2')
                    T_curr += (-(U_value * np.pi * D / (m_dot * cp)) * (T_curr - T_amb_array[i])) * dx
                except: break
            
            # Score berechnen (Energie + Gewichtung pro Hardware-Station)
            score = total_energy + (station_count * 1e6) 
            
            if score < best_score:
                best_score = score
                best_start_T = T_start_test
                best_zone_params = (z_min, z_max)

    # --- Finale Simulation des Siegers ---
    T_curr, P_curr = best_start_T + 273.15, P_inlet
    z_min, z_max = best_zone_params
    distances, temperatures, pressures = [0.0], [T_curr], [P_curr]
    
    for i in range(n_steps):
        try:
            T_c = T_curr - 273.15
            if (P_curr < 100.5e5) or (T_c < z_min) or (T_c > z_max):
                P_curr += max_boost
                T_curr = best_start_T + 273.15
                # Stationen im Druck-Plot markieren
                ax2.plot(i*dx/1000, P_curr/1e5, 'o', color=current_color, markersize=4, alpha=0.4)

            rho = PropsSI('D', 'T', T_curr, 'P', P_curr, 'CO2')
            cp = PropsSI('C', 'T', T_curr, 'P', P_curr, 'CO2')
            mu = PropsSI('V', 'T', T_curr, 'P', P_curr, 'CO2')
            v = m_dot / (rho * (np.pi * (D/2)**2))
            Re = (rho * v * D) / mu
            f = 0.25 / (np.log10(epsilon / (3.7 * D) + 5.74 / (Re**0.9)))**2
            
            # Joule-Thomson Effekt
            h_ref = PropsSI('H', 'T', T_curr, 'P', P_curr, 'CO2')
            T_jt_check = PropsSI('T', 'H', h_ref, 'P', P_curr - 1e5, 'CO2')
            mu_jt = (T_jt_check - T_curr) / (-1e5)

            dp = (-f * (rho * v**2) / (2 * D)) * dx
            P_curr += dp
            T_curr += (-(U_value * np.pi * D / (m_dot * cp)) * (T_curr - T_amb_array[i]) + mu_jt * (dp/dx)) * dx
            
            pressures.append(P_curr); temperatures.append(T_curr); distances.append((i+1)*dx)
        except: break

    # Kurven zeichnen
    label_str = f'D={D}m (T_opt={best_start_T}°C)'
    ax1.plot(np.array(distances)/1000, np.array(temperatures)-273.15, color=current_color, label=label_str, lw=2)
    ax2.plot(np.array(distances)/1000, np.array(pressures)/1e5, color=current_color, lw=2)
    print(f"Berechnet: D={D}m -> Strategie: {'Zone 1' if best_start_T < 30 else 'Zone 2'} ({best_start_T}°C)")

# --- 3. Formatierung & Legende ---

# Temperatur-Diagramm
ax1.plot(dist_full_km, T_amb_array-273.15, 'k:', alpha=0.5, label='Boden (T_amb)')
ax1.set_ylabel('Temperatur (°C)')
ax1.set_title('Optimierte CO2-Pipeline: Vergleich der Durchmesser')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Konfigurationen")
ax1.grid(True, alpha=0.2)

# Druck-Diagramm
ax2.axhline(100, color='grey', linestyle='--', alpha=0.5, label='Grenze 100 bar')
# Synchronisation der Legende: Wir übernehmen die Handles von ax1
handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Legende (Druck)")

ax2.set_ylabel('Druck (bar)')
ax2.set_xlabel('Distanz (km)')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()