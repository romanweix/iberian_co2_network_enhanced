import matplotlib.pyplot as plt
from neqsim import jneqsim

# 1. Thermodynamics Setup (Überkritisches CO2: 50°C, 150 bar)
fluid = jneqsim.thermo.system.SystemPrEos(273.15 + 50.0, 150.0)
fluid.addComponent("CO2", 1.0)
fluid.setMixingRule("classic")

# Flash-Berechnung ausführen
jneqsim.thermodynamicoperations.ThermodynamicOperations(fluid).TPflash()

# 2. Process Setup
# KORREKTE Reihenfolge: erst Name (String), dann fluid (SystemInterface)
inlet = jneqsim.process.equipment.stream.Stream("CO2 Inlet", fluid)
inlet.setFlowRate(100.0, "kg/sec")

num_legs = 50
pipe = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills(
    "CO2 Pipeline", inlet
)
pipe.setLength(150000.0)  # 150 km in Meter
pipe.setDiameter(0.4)      # 40 cm Innendurchmesser
pipe.setOuterHeatTransferCoefficient(5.0)
pipe.setNumberOfLegs(num_legs)
pipe.setOuterTemperatures([273.15 + 15.0] * num_legs)  # 15°C Umgebungstemperatur

# 3. Solve System
process = jneqsim.process.processmodel.ProcessSystem()
process.add(inlet)
process.add(pipe)
process.run()

# 4. Extract Node Profiles
# Direkter Zugriff auf die integrierte Ergebnis-Matrix von NeqSim
# Spalten in profile: [0] Distance (m), [1] Pressure (bar), [2] Temperature (K)
profile = pipe.getProfileResults()

length_km = [row[0] / 1000.0 for row in profile]
pressures_bar = [row[1] for row in profile]
temperatures_c = [row[2] - 273.15 for row in profile]

# 5. Plot Results
fig, ax1 = plt.subplots(figsize=(10, 5))

# Druck-Profil (linke Y-Achse)
ax1.plot(length_km, pressures_bar, color="tab:red", lw=2, label="Druck (bar)")
ax1.set_xlabel("Pipeline-Länge (km)")
ax1.set_ylabel("Druck (bar)", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.grid(True, linestyle="--", alpha=0.5)

# Temperatur-Profil (rechte Y-Achse)
ax2 = ax1.twinx()
ax2.plot(
    length_km,
    temperatures_c,
    color="tab:blue",
    lw=2,
    linestyle="--",
    label="Temperatur (°C)",
)
ax2.set_ylabel("Temperatur (°C)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("NeqSim CO2-Pipeline Profile (150 km)")
fig.tight_layout()
plt.show()