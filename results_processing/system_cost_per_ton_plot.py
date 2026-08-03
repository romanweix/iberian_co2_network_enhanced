import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline

# -----------------------------
# User config
# -----------------------------
BASE_DIR = Path("results_summaries")
FILE_TEMPLATE = "stochastic_results_theta_{theta:.2f}.xlsx"
SCENARIO = "HUS"               # Choose between "LUS", "EUS" and "HUS"

# Amount of CO2 captured at alpha=1 (in Mt)
TOTAL_CO2_ALPHA1_MT = 3028.0

# Whether to include Injection in the transport cost
INCLUDE_INJECTION = False

# Whether to include initial compression costs (CAPEX, OPEX, Electricity)
INCLUDE_INIT_COMPR = False  # set to False to exclude initial compression costs

# Flexible match helper (case-insensitive, trims)
def _ci(name):
    return str(name).strip().lower()

# --- Raw labels ---
RAW_LABELS = {
    "injection": "Injection cost",
    "capex_onshore": "CAPEX onshore pipe",
    "capex_offshore": "CAPEX offshore pipe",
    "opex_onshore": "OPEX onshore pipe",
    "opex_offshore": "OPEX offshore pipe",
    "capex_boost_init": "CAPEX initial boosting stations",   # initial compression CAPEX
    "capex_boost_add":  "CAPEX additional boosting stations",# boosters CAPEX
    "opex_boost_init":  "OPEX initial boosting stations",    # initial compression OPEX
    "opex_boost_add":   "OPEX additional boosting stations", # boosters OPEX
    "elec_boost_init":  "Electricity cost initial boosting stations",
    "elec_boost_add":   "Electricity cost additional boosting stations",
}

# Aggregated categories
AGG_ORDER = [
    # "Injection",
    "Pipeline CAPEX",
    "Pipeline OPEX",
    # "Initial compression CAPEX",
    # "Initial compression OPEX",
    "Boosters CAPEX",
    "Boosters OPEX",
    # "Electricity (initial)",
    "Electricity (boosters)",
]

# Pastel color palette
PASTEL = {
    "Injection": "#DD9FAE",
    "Pipeline CAPEX": "#E297FF",
    "Pipeline OPEX": "#CAB2D6",
    "Initial compression CAPEX": "#FFD2B3",
    "Initial compression OPEX": "#D9B98C",
    "Boosters CAPEX": "#F5F578",
    "Boosters OPEX": "#D5D7B3",
    "Electricity (initial)": "#99B4FF",
    "Electricity (boosters)": "#6FD9FF",
}

# -----------------------------
# Helpers
# -----------------------------
def get_total_col(df: pd.DataFrame) -> str:
    """Return the name of the 'Total' column, tolerant to variants."""
    for c in df.columns:
        if _ci(c).startswith("total"):
            return c
    raise KeyError("Couldn't find a 'Total' column in the sheet.")

def read_sheet(theta: float) -> pd.DataFrame:
    file_path = BASE_DIR / FILE_TEMPLATE.format(theta=theta)
    sheet_name = f"{SCENARIO} - Cost breakdown"
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    if "Concept" not in df.columns:
        df.rename(columns={df.columns[0]: "Concept"}, inplace=True)
    df["Concept"] = df["Concept"].astype(str).str.strip()
    return df

def pick(df_idx, label):
    """Pick a value by matching the label case-insensitively; 0 if missing."""
    if label in df_idx.index:
        return pd.to_numeric(df_idx.at[label, df_idx.columns[0]], errors="coerce") or 0.0
    for idx in df_idx.index:
        if _ci(idx) == _ci(label):
            return pd.to_numeric(df_idx.at[idx, df_idx.columns[0]], errors="coerce") or 0.0
    return 0.0

def read_pipes_bs(theta: float) -> pd.DataFrame:
    file_path = BASE_DIR / FILE_TEMPLATE.format(theta=theta)
    df = pd.read_excel(file_path, sheet_name="EUS - Pipes")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, target: str):
    """Case-insensitive column finder."""
    target_ci = _ci(target)
    for c in df.columns:
        if _ci(c) == target_ci:
            return c
    for c in df.columns:
        if target_ci in _ci(c):
            return c
    raise KeyError(f"Column '{target}' not found in '{SCENARIO} - PIPES'.")

def compute_boost_scalars(df_bs: pd.DataFrame) -> tuple[float, float]:
    """
    Returns:
      - n_boosters: sum of 'Number of boosters' (unitless)
      - total_boosted: sum over rows with booster==1 of (sum of flows in 2030..2050)*5  [Mt]
    """
    col_nb = _find_col(df_bs, "Number of boosters")
    flow_targets = ["Flow in 2030", "Flow in 2035", "Flow in 2040", "Flow in 2045", "Flow in 2050"]
    flow_cols = [_find_col(df_bs, t) for t in flow_targets]

    tmp = df_bs.copy()
    tmp[col_nb] = pd.to_numeric(tmp[col_nb], errors="coerce").fillna(0.0)
    for c in flow_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)

    n_boosters = float(tmp[col_nb].sum())

    rows = tmp[tmp[col_nb] == 1]
    total_boosted = float((rows[flow_cols].sum(axis=1) * 5.0).sum())  # 5 years per step

    return n_boosters, total_boosted

# -----------------------------
# Build dataset (€/t) vs transported CO2 (Mt)
# -----------------------------
thetas = [i/100 for i in range(0, 101, 10)]  # 0.00 .. 1.00
records = []

for th in thetas:
    transported_mt = TOTAL_CO2_ALPHA1_MT * th  # Mt transported at this theta
    if transported_mt <= 0:
        continue  # LCO undefined at 0

    # --- Read cost breakdown (M€)
    df = read_sheet(th)
    total_col = get_total_col(df)
    df_idx = df.set_index("Concept")[[total_col]].copy()
    df_idx.columns = ["Total"]

    injection = pick(df_idx, RAW_LABELS["injection"])
    capex_on  = pick(df_idx, RAW_LABELS["capex_onshore"])
    capex_off = pick(df_idx, RAW_LABELS["capex_offshore"])
    opex_on   = pick(df_idx, RAW_LABELS["opex_onshore"])
    opex_off  = pick(df_idx, RAW_LABELS["opex_offshore"])

    capex_ini = pick(df_idx, RAW_LABELS["capex_boost_init"])   # M€
    capex_add = pick(df_idx, RAW_LABELS["capex_boost_add"])    # M€
    opex_ini  = pick(df_idx, RAW_LABELS["opex_boost_init"])    # M€
    opex_add  = pick(df_idx, RAW_LABELS["opex_boost_add"])     # M€
    elec_ini  = pick(df_idx, RAW_LABELS["elec_boost_init"])    # M€
    elec_add  = pick(df_idx, RAW_LABELS["elec_boost_add"])     # M€

    # --- Boosters patch: scale CAPEX/OPEX by total boosted mass over horizon
    df_bs = read_pipes_bs(th)
    n_boosters, total_boosted = compute_boost_scalars(df_bs)   # n (unitless), Mt (horizon)

    # Optionally exclude initial compression
    if not INCLUDE_INIT_COMPR:
        capex_ini = 0.0
        opex_ini  = 0.0
        elec_ini  = 0.0

    # Aggregations (still M€)
    pipeline_capex = capex_on + capex_off
    pipeline_opex  = opex_on + opex_off

    # Convert to €/t by dividing by transported Mt (million €/Mt → €/t)
    to_eur_per_t = lambda x: x / transported_mt
    rec = {
        "transported_mt": transported_mt,
        "Injection": to_eur_per_t(injection),
        "Pipeline CAPEX": to_eur_per_t(pipeline_capex),
        "Pipeline OPEX": to_eur_per_t(pipeline_opex),
        "Initial compression CAPEX": to_eur_per_t(capex_ini),
        "Initial compression OPEX": to_eur_per_t(opex_ini),
        "Boosters CAPEX": to_eur_per_t(capex_add),
        "Boosters OPEX": to_eur_per_t(opex_add),
        "Electricity (initial)": to_eur_per_t(elec_ini),
        "Electricity (boosters)": to_eur_per_t(elec_add),
    }

    if not INCLUDE_INJECTION:
        rec.pop("Injection", None)

    # Total LCO (€/t) = sum of all cost components (no revenues)
    rec["TOTAL_LCO"] = sum(rec[k] for k in AGG_ORDER if k in rec)

    records.append(rec)

res = pd.DataFrame(records).sort_values("transported_mt").reset_index(drop=True)

# -----------------------------
# Plot (stacked bars €/t)
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 6))

x = res["transported_mt"].values  # Mt
step = np.min(np.diff(x)) if len(x) > 1 else 100.0
width = 0.6 * step

bottom = np.zeros(len(res))
for label in AGG_ORDER:
    if label not in res.columns:
        continue
    y = res[label].values  # €/t
    ax.bar(x, y, width=width, bottom=bottom, label=label,
           color=PASTEL.get(label, None), edgecolor="white", linewidth=0.6, zorder=3)
    bottom += y

# Axes, limits, ticks
ax.set_xlabel("CO₂ transported (Mt)")
ax.set_ylabel("Levelized Cost of CO₂ Transport (€/tCO₂)")
ax.set_xlim(150, 3180)
ax.set_ylim(0, 2.10)
ax.set_xticks(x)
ax.set_xticklabels([f"{mt:.0f}" for mt in x])


# -----------------------------
# Secondary x-axis: θ
# -----------------------------
def x_to_theta(x):
    return x / TOTAL_CO2_ALPHA1_MT  # from Mt -> θ

def theta_to_x(theta):
    return theta * TOTAL_CO2_ALPHA1_MT  # from θ -> Mt

secax = ax.secondary_xaxis('top', functions=(x_to_theta, theta_to_x))
secax.set_xlabel("θ (fraction of CO₂ captured)")

# Ticks of 𝜃 aligned with the bars
theta_vals = res["transported_mt"].values / TOTAL_CO2_ALPHA1_MT
secax.set_xticks(theta_vals)
secax.set_xticklabels([f"{t:.1f}" for t in theta_vals])


ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98),
                ncol=3, frameon=True)
leg._legend_box.align = "left"
leg.get_frame().set_edgecolor("black")

plt.tight_layout()
outname = f"transport_cost_per_ton_plot_{SCENARIO}.png"
outpath = Path("_cost_breakdown_per_ton_plots")
outpath.mkdir(exist_ok=True)
fig.savefig(outpath / outname, dpi=300, bbox_inches="tight")
print(f"✔ Saved figure: {outpath / outname}")

plt.show()
