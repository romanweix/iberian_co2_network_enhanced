import numpy as np
import matplotlib.pyplot as plt

# --- Parameter ---
L = 50000          # 50 km
D = 0.4            
v = 2.0            
p_start = 100.0    
T_start = 15.0     # Starttemperatur CO2 (°C)
T_ground = 8.0     # Bodentemperatur (°C)
k_wall = 2.0       # Wärmedurchgangskoeffizient (W/m²K) - Schätzwert für isolierte Rohre

# --- Diskretisierung ---
n_steps = 500
x = np.linspace(0, L, n_steps)
dx = L / n_steps
h = np.linspace(0, 200, n_steps) # Höhenprofil +200m

# --- Arrays initialisieren ---
pressure = np.zeros(n_steps)
temp = np.zeros(n_steps)
density = np.zeros(n_steps)

pressure[0] = p_start
temp[0] = T_start

# --- Simulation ---
for i in range(1, n_steps):
    # 1. Temperaturänderung (Vereinfachter Wärmetausch)
    # Massenstrom m_dot = rho * A * v
    area = np.pi * (D/2)**2
    current_rho = 900 + (15 - temp[i-1]) * 4 # Grobe Dichteänderung: CO2 wird dichter wenn kälter
    m_dot = current_rho * area * v
    cp = 2000 # Spezifische Wärmekapazität CO2 (J/kgK)
    
    # Wärmeübergang Q = k * Fläche * deltaT
    dQ = k_wall * (np.pi * D * dx) * (T_ground - temp[i-1])
    dT = dQ / (m_dot * cp)
    temp[i] = temp[i-1] + dT
    
    # 2. Dichte-Update (Dense Phase Näherung)
    # Bei 100 bar: 15°C -> ~900 kg/m³, 8°C -> ~940 kg/m³
    density[i] = 900 + (15 - temp[i]) * 5.7 

    # 3. Druckverlust (Reibung + Höhe)
    dp_friction = 0.015 * (dx / D) * (density[i] / 2) * (v**2) / 1e5
    dh = h[i] - h[i-1]
    dp_static = (density[i] * 9.81 * dh) / 1e5
    
    pressure[i] = pressure[i-1] - dp_friction - dp_static

# --- Plotten ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# Druck und Temperatur Achsen
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))

# Plots
p1, = ax1.plot(x/1000, pressure, 'b-', label="Druck (bar)", linewidth=2)
p2, = ax2.plot(x/1000, temp, 'r--', label="Temperatur (°C)")
ax3.fill_between(x/1000, 0, h, color='gray', alpha=0.2, label="Höhenprofil")

# Achsen-Labels
ax1.set_xlabel("Distanz (km)")
ax1.set_ylabel("Druck (bar)", color='b')
ax2.set_ylabel("Temperatur (°C)", color='r')
ax3.set_ylabel("Höhe (m)", color='gray')

ax1.axhline(y=73.8, color='darkred', linestyle=':', label="Kritischer Druck")
plt.title("CO2-Pipeline: Kopplung von Druck, Temperatur und Höhe")
ax1.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.show()