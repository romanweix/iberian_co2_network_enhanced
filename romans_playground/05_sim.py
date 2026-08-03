import numpy as np
import matplotlib.pyplot as plt

def simulate_co2_pipeline():
    # --- Parameter ---
    L = 500_000          # Gesamtlänge in m
    dx = 1000            # Schrittweite in m
    steps = int(L / dx)
    
    D = 0.5              # Durchmesser Pipeline in m (500mm)
    rho = 800            # Durchschnittliche Dichte überkritisches CO2 (kg/m^3)
    viscosity = 0.00006  # Dynamische Viskosität (Pa*s)
    velocity = 1.5       # Fließgeschwindigkeit (m/s)
    roughness = 0.000045 # Rohrrauhigkeit (m)
    U_value = 2.0        # Wärmedurchgangskoeffizient zum Boden (W/m^2K)
    cp_co2 = 2000        # Spezifische Wärmekapazität (J/kgK)
    g = 9.81
    
    # Startwerte
    P = 150.0            # bar
    T = 50.0             # °C
    T_ambient = 0.0      # Bodentemperatur in °C
    
    # Topologie-Profil (Beispiel: Sinuskurve für Hügel)
    distances = np.linspace(0, L, steps)
    elevations = 200 * np.sin(2 * np.pi * distances / 100_000) # 200m Schwankung
    
    p_results = [P]
    t_results = [T]
    
    # --- Simulation ---
    current_p = P
    current_t = T
    
    for i in range(1, steps):
        # 1. Druckverlust durch Reibung (vereinfacht)
        # Darcy-Weisbach: dp = f * (L/D) * (rho * v^2 / 2)
        f = 0.015 # Reibungsbeiwert (geschätzt)
        dp_friction = f * (dx / D) * (rho * velocity**2 / 2) / 1e5 # in bar
        
        # 2. Druckänderung durch Höhe (Hydrostatischer Druck)
        dh = elevations[i] - elevations[i-1]
        dp_gravity = -(rho * g * dh) / 1e5 # in bar
        
        current_p += (dp_gravity - dp_friction)
        
        # 3. Temperaturänderung (Wärmeübergang an Umgebung)
        # Q = U * A * deltaT
        surface_area = np.pi * D * dx
        mass_flow = rho * velocity * (np.pi * (D/2)**2)
        
        # dT = Q / (mass_flow * cp)
        dT = - (U_value * surface_area * (current_t - T_ambient)) / (mass_flow * cp_co2)
        current_t += dT
        
        p_results.append(current_p)
        t_results.append(current_t)

    # --- Visualisierung ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Distanz (km)')
    ax1.set_ylabel('Druck (bar)', color='tab:blue')
    ax1.plot(distances/1000, p_results, color='tab:blue', label='Druck')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperatur (°C) / Höhe (m)', color='tab:red')
    ax2.plot(distances/1000, t_results, color='tab:red', label='Temperatur')
    ax2.plot(distances/1000, elevations/10, color='black', alpha=0.3, label='Höhe (skaliert)')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('CO2-Pipeline Simulation (500km)')
    fig.tight_layout()
    plt.show()

simulate_co2_pipeline()