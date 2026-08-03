# -*- coding: utf-8 -*-
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
    "capex_boost_init": "CAPEX initial boosting stations",       # Initial compression CAPEX
    "capex_boost_add": "CAPEX additional boosting stations",     # Boosters CAPEX
    "opex_boost_init": "OPEX initial boosting stations",         # Initial compression OPEX
    "opex_boost_add": "OPEX additional boosting stations",       # Boosters OPEX
    "elec_boost_init": "Electricity cost initial boosting stations",
    "elec_boost_add": "Electricity cost additional boosting stations",
    "shipping": "Shipping cost",
    "revenue": "CO2 sales revenue",
}

# --- Aggregated categories for plotting ---
AGG_ORDER = [
    "Injection",
    "Pipeline CAPEX",
    "Pipeline OPEX",
    "Initial compression CAPEX",
    "Initial compression OPEX",
    "Boosters CAPEX",
    "Boosters OPEX",
    "Electricity (initial)",
    "Electricity (boosters)",
    "Shipping",
    "CO₂ sales revenue",
]

# Pastel palette (soft tones)
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
    "Shipping": "#918367",
    "CO₂ sales revenue": "#B4E9A9",
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
    """Read the Cost breakdown sheet for a given theta."""
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
    """Read the 'EUS - Pipes' sheet robustly."""
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
    # allow mild variants like commas vs dots or extra spaces
    for c in df.columns:
        if target_ci in _ci(c):
            return c
    raise KeyError(f"Column '{target}' not found in 'Pipes - BS'.")

def compute_boost_scalars(df_bs: pd.DataFrame) -> tuple[float, float]:
    """
    Compute:
      - n_boosters = sum of 'Number of boosters'
      - total_boosted = sum over rows with booster==1 of (sum of flows in 2030..2050)*5
    Returns (n_boosters, total_boosted) both in Mt for consistency with the workbook.
    """
    col_nb = _find_col(df_bs, "Number of boosters")
    # flow columns (ensure exact names used in the workbook)
    flow_cols_targets = ["Flow in 2030", "Flow in 2035", "Flow in 2040", "Flow in 2045", "Flow in 2050"]
    flow_cols = [ _find_col(df_bs, t) for t in flow_cols_targets ]

    # Clean numeric
    tmp = df_bs.copy()
    tmp[col_nb] = pd.to_numeric(tmp[col_nb], errors="coerce").fillna(0.0)
    for c in flow_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)

    n_boosters = float(tmp[col_nb].sum())

    # only rows with booster==1
    rows = tmp[tmp[col_nb] == 1]
    # sum flows per row across years
    per_row = rows[flow_cols].sum(axis=1) * 5.0  # multiply by 5 years per step
    total_boosted = float(per_row.sum())

    return n_boosters, total_boosted

# -----------------------------
# Build dataset across thetas
# -----------------------------
thetas = [i/100 for i in range(0, 101, 10)]  # 0.00 .. 1.00
records = []

for th in thetas:
    # --- Costs sheet
    df = read_sheet(th)
    total_col = get_total_col(df)
    df_idx = df.set_index("Concept")[[total_col]].copy()
    df_idx.columns = ["Total"]

    # Read raw components (million € in the workbook)
    injection = pick(df_idx, RAW_LABELS["injection"])
    capex_on = pick(df_idx, RAW_LABELS["capex_onshore"])
    capex_off = pick(df_idx, RAW_LABELS["capex_offshore"])
    opex_on = pick(df_idx, RAW_LABELS["opex_onshore"])
    opex_off = pick(df_idx, RAW_LABELS["opex_offshore"])

    capex_init = pick(df_idx, RAW_LABELS["capex_boost_init"])  # Initial compression CAPEX (M€)
    capex_add  = pick(df_idx, RAW_LABELS["capex_boost_add"])   # Boosters CAPEX (M€)
    opex_init  = pick(df_idx, RAW_LABELS["opex_boost_init"])   # Initial compression OPEX (M€)
    opex_add   = pick(df_idx, RAW_LABELS["opex_boost_add"])    # Boosters OPEX (M€)

    elec_ini = pick(df_idx, RAW_LABELS["elec_boost_init"])     # Electricity initial (M€)
    elec_add = pick(df_idx, RAW_LABELS["elec_boost_add"])      # Electricity boosters (M€)
    shipping = pick(df_idx, RAW_LABELS["shipping"])
    revenue_raw = pick(df_idx, RAW_LABELS["revenue"])

    # --- Boosters “patch”: scale CAPEX/OPEX by total boosted mass over horizon
    df_bs = read_pipes_bs(th)
    n_boosters, total_boosted = compute_boost_scalars(df_bs)  # n (unitless) and Mt (over horizon)

    # # Avoid division by zero
    # if n_boosters > 0 and total_boosted > 0:
    #     # Convert to per-booster-per-ton (M€/Mt) and then multiply by boosted Mt over horizon.
    #     capex_add_corr = (capex_add / n_boosters) * total_boosted
    #     opex_add_corr  = (opex_add  / n_boosters) * total_boosted
    # else:
    #     capex_add_corr = 0.0
    #     opex_add_corr  = 0.0

    # Aggregations
    pipeline_capex = capex_on + capex_off
    pipeline_opex = opex_on + opex_off
    revenue_mag = abs(revenue_raw)  # ensure positive magnitude

    # Convert to billions (sheet is in million €)
    to_b = lambda x: x / 1000.0
    rec = {
        "theta": round(th, 2),
        "Injection": to_b(injection),
        "Pipeline CAPEX": to_b(pipeline_capex),
        "Pipeline OPEX": to_b(pipeline_opex),
        "Initial compression CAPEX": to_b(capex_init),
        "Initial compression OPEX": to_b(opex_init),
        "Boosters CAPEX": to_b(capex_add),
        "Boosters OPEX": to_b(opex_add),
        "Electricity (initial)": to_b(elec_ini),
        "Electricity (boosters)": to_b(elec_add),
        "Shipping": to_b(shipping),
        "CO₂ sales revenue": to_b(revenue_mag),
    }

    # Net cost = sum(costs) - revenue
    cost_sum = sum(rec[k] for k in AGG_ORDER if k != "CO₂ sales revenue")
    rec["NET"] = cost_sum - rec["CO₂ sales revenue"]
    records.append(rec)

res = pd.DataFrame(records).set_index("theta").sort_index()


# -----------------------------
# Results table for θ = 1.00
# -----------------------------
theta_target = 1.00

theta_row = res.index.to_series().sub(theta_target).abs().idxmin()
row = res.loc[theta_row]

# Equal order as in the plot legend
cats = [
    "Injection",
    "Pipeline CAPEX",
    "Pipeline OPEX",
    "Initial compression CAPEX",
    "Initial compression OPEX",
    "Boosters CAPEX",
    "Boosters OPEX",
    "Electricity (initial)",
    "Electricity (boosters)",
    "Shipping",
    "CO₂ sales revenue",
]

# Build output DataFrame (includes the black line metric: Net cost)
out = pd.DataFrame({
    "Category": ["Net cost (cost − revenue)"] + cats,
    "Value (billion €)": [row["NET"]] + [row[c] for c in cats],
}).set_index("Category")

# Round for better readability
out = out.astype(float).round(2)

# print(f"Results for θ = {theta_row:.2f}")
# print(out)


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 6))

x = res.index.values
width = 0.06  # thin bars

# Upward stacks (costs)
bottom = np.zeros(len(res))
for label in AGG_ORDER:
    if label == "CO₂ sales revenue":
        continue
    y = res[label].values
    ax.bar(x, y, width=width, bottom=bottom, label=label,
           color=PASTEL.get(label, None), edgecolor="white", linewidth=0.6, zorder=3)
    bottom += y

# Downward revenue
rev = -res["CO₂ sales revenue"].values
ax.bar(x, rev, width=width, bottom=np.zeros(len(res)), label="CO₂ sales revenue",
       color=PASTEL.get("CO₂ sales revenue", None), edgecolor="white", linewidth=0.6, zorder=3)

# Smooth cubic curve through NET points
x_net = x
y_net = res["NET"].values
x_dense = np.linspace(x_net.min(), x_net.max(), 500)
cs = CubicSpline(x_net, y_net)
y_dense = cs(x_dense)

ax.plot(x_dense, y_dense, color="black", linewidth=1.5,
        label="Net cost (cost − revenue)", zorder=4)
ax.scatter(x_net, y_net, color="black", s=20, zorder=5)

# Axes, limits, ticks
ax.set_xlabel("θ (fraction of CO₂ captured)")
ax.set_ylabel("Costs for the system operator (billion €)")
ax.set_xlim(-0.03, 1.05)
ax.set_ylim(-50, 67)
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in x])

# Grid and legend
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98),
                ncol=3, frameon=True)
leg._legend_box.align = "left"
frame = leg.get_frame()
frame.set_edgecolor("black")

plt.tight_layout()


outname = f"operator_cost_breakdown_vs_theta_{SCENARIO}.png"
outpath = Path("operator_costs_breakdown")
outpath.mkdir(exist_ok=True)
fig.savefig(outpath / outname, dpi=300, bbox_inches="tight")
print(f"✔ Saved figure: {outpath / outname}")

plt.show()
