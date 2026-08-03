import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

# --------------------------------------------------
# User config
# --------------------------------------------------
BASE_DIR = Path("results_summaries")

SCENARIO = "EUS"               # Choose between "LUS", "EUS" and "HUS"
THETA = 1.00                   # Choose between 0.90 and 1.00

TRAD_FILE = BASE_DIR / f"stochastic_traditional_results_theta_{THETA:.2f}.xlsx"
DEV_FILE  = BASE_DIR / f"stochastic_results_theta_{THETA:.2f}.xlsx"

# --------------------------------------------------
# Labels and helper functions
# --------------------------------------------------
def _ci(name):  # case-insensitive normalization
    return str(name).strip().lower()

RAW_LABELS = {
    # "injection": "Injection cost",
    "capex_onshore": "CAPEX onshore pipe",
    "capex_offshore": "CAPEX offshore pipe",
    "opex_onshore": "OPEX onshore pipe",
    "opex_offshore": "OPEX offshore pipe",
    # "capex_boost_init": "CAPEX initial boosting stations",
    "capex_boost_add": "CAPEX additional boosting stations",
    # "opex_boost_init": "OPEX initial boosting stations",
    "opex_boost_add": "OPEX additional boosting stations",
    # "elec_boost_init": "Electricity cost initial boosting stations",
    "elec_boost_add": "Electricity cost additional boosting stations",
    "shipping": "Shipping cost",
    "revenue": "CO2 sales revenue",
}

# Categories to be plotted
PLOT_CATS = [
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

# Color palette
PASTEL = {
    # "Injection": "#DD9FAE",
    "Pipeline CAPEX": "#E297FF",
    "Pipeline OPEX": "#CAB2D6",
    # "Initial compression CAPEX": "#FFD2B3",
    # "Initial compression OPEX": "#D9B98C",
    "Boosters CAPEX": "#F5F578",
    "Boosters OPEX": "#D5D7B3",
    # "Electricity (initial)": "#99B4FF",
    "Electricity (boosters)": "#6FD9FF",
    "Shipping": "#918367",
    "CO₂ sales revenue": "#B4E9A9",
}

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def get_total_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if _ci(c).startswith("total"):
            return c
    raise KeyError("Couldn't find a 'Total' column in the sheet.")

def _find_col(df: pd.DataFrame, target: str):
    t = _ci(target)
    for c in df.columns:
        if _ci(c) == t:
            return c
    for c in df.columns:
        if t in _ci(c):
            return c
    raise KeyError(f"Column '{target}' not found.")

def pick(df_idx, label):
    """Returns the numeric value for a given label (case-insensitive)."""
    if label in df_idx.index:
        return pd.to_numeric(df_idx.at[label, df_idx.columns[0]], errors="coerce") or 0.0
    for idx in df_idx.index:
        if _ci(idx) == _ci(label):
            return pd.to_numeric(df_idx.at[idx, df_idx.columns[0]], errors="coerce") or 0.0
    return 0.0

def pick_any(df_idx: pd.DataFrame, labels: list[str]) -> float:
    """Returns the first existing label value (case-insensitive)."""
    for lbl in labels:
        for idx in df_idx.index:
            if _ci(idx) == _ci(lbl):
                val = pd.to_numeric(df_idx.at[idx, df_idx.columns[0]], errors="coerce")
                return float(0.0 if pd.isna(val) else val)
    for lbl in labels:
        for idx in df_idx.index:
            if _ci(lbl) in _ci(idx):
                val = pd.to_numeric(df_idx.at[idx, df_idx.columns[0]], errors="coerce")
                return float(0.0 if pd.isna(val) else val)
    return 0.0

def compute_boost_scalars(df_bs: pd.DataFrame) -> tuple[float, float]:
    """Returns (n_boosters, total_boosted) in Mt for the time horizon."""
    try:
        col_nb = _find_col(df_bs, "Number of boosters")
    except KeyError:
        col_nb = _find_col(df_bs, "Traditional boosters (150 km rule)")

    flow_targets = ["Flow in 2030", "Flow in 2035", "Flow in 2040", "Flow in 2045", "Flow in 2050"]
    flow_cols = [_find_col(df_bs, t) for t in flow_targets]

    tmp = df_bs.copy()
    tmp[col_nb] = pd.to_numeric(tmp[col_nb], errors="coerce").fillna(0.0)
    for c in flow_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)

    n_boosters = float(tmp[col_nb].sum())
    rows = tmp[tmp[col_nb] == 1]
    total_boosted = float((rows[flow_cols].sum(axis=1) * 5.0).sum())  # 5 years per segment

    return n_boosters, total_boosted

def read_costs(file_path: Path) -> dict:
    """Reads cost breakdown and booster data from an Excel file."""
    df = pd.read_excel(file_path, sheet_name=f"{SCENARIO} - Cost breakdown")
    df.columns = [str(c).strip() for c in df.columns]
    if "Concept" not in df.columns:
        df.rename(columns={df.columns[0]: "Concept"}, inplace=True)
    df["Concept"] = df["Concept"].astype(str).str.strip()
    total_col = get_total_col(df)
    df_idx = df.set_index("Concept")[[total_col]].copy()
    df_idx.columns = ["Total"]

    # injection = pick(df_idx, RAW_LABELS["injection"])
    capex_on  = pick(df_idx, RAW_LABELS["capex_onshore"])
    capex_off = pick(df_idx, RAW_LABELS["capex_offshore"])
    opex_on   = pick(df_idx, RAW_LABELS["opex_onshore"])
    opex_off  = pick(df_idx, RAW_LABELS["opex_offshore"])
    # capex_ini = pick_any(df_idx, ALIAS["capex_boost_init"])
    capex_add = pick_any(df_idx, ALIAS["capex_boost_add"])
    # opex_ini  = pick_any(df_idx, ALIAS["opex_boost_init"])
    opex_add  = pick_any(df_idx, ALIAS["opex_boost_add"])
    # elec_ini  = pick_any(df_idx, ALIAS["elec_boost_init"])
    elec_add  = pick_any(df_idx, ALIAS["elec_boost_add"])

    df_bs = pd.read_excel(file_path, sheet_name=f"{SCENARIO} - Pipes")
    df_bs.columns = [str(c).strip() for c in df_bs.columns]
    n_boosters, total_boosted = compute_boost_scalars(df_bs)

    pipeline_capex = capex_on + capex_off
    pipeline_opex  = opex_on + opex_off

    to_b = lambda x: x / 1000.0  # from M€ to B€
    return {
        # "Injection": to_b(injection),
        "Pipeline CAPEX": to_b(pipeline_capex),
        "Pipeline OPEX": to_b(pipeline_opex),
        # "Initial compression CAPEX": to_b(capex_ini),
        # "Initial compression OPEX": to_b(opex_ini),
        "Boosters CAPEX": to_b(capex_add),
        "Boosters OPEX": to_b(opex_add),
        # "Electricity (initial)": to_b(elec_ini),
        "Electricity (boosters)": to_b(elec_add),
    }

ALIAS = {
    "capex_boost_init": [
        "CAPEX initial boosting stations",
        "CAPEX initial compression stations",
    ],
    "opex_boost_init": [
        "OPEX initial boosting stations",
        "OPEX initial compression stations",
    ],
    "elec_boost_init": [
        "Electricity cost initial boosting stations",
        "Electricity cost initial compression stations",
    ],
    "capex_boost_add": [
        "CAPEX additional boosting stations",
        "CAPEX traditional boosters",
    ],
    "opex_boost_add": [
        "OPEX additional boosting stations",
        "OPEX traditional boosters",
    ],
    "elec_boost_add": [
        "Electricity cost additional boosting stations",
        "Electricity cost traditional boosters",
    ],
}

# --------------------------------------------------
# Load data for both models
# --------------------------------------------------
trad = read_costs(TRAD_FILE)
dev  = read_costs(DEV_FILE)

# Comparative DataFrame
df_cmp = pd.DataFrame({
    "Traditional (B€)": [trad[c] for c in PLOT_CATS],
    f"Developed θ={THETA:.2f} (B€)": [dev[c] for c in PLOT_CATS],
}, index=PLOT_CATS)

df_cmp["Δ Dev−Trad (B€)"] = df_cmp[f"Developed θ={THETA:.2f} (B€)"] - df_cmp["Traditional (B€)"]
df_cmp["Ratio Dev/Trad"]  = np.where(df_cmp["Traditional (B€)"] != 0,
                                     df_cmp[f"Developed θ={THETA:.2f} (B€)"] / df_cmp["Traditional (B€)"],
                                     np.nan)

# --------------------------------------------------
# Plot: grouped bars (same colors by category)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(PLOT_CATS))
width = 0.38

# Traditional model bars
bars_trad = ax.bar(
    x - width/2,
    df_cmp["Traditional (B€)"].values,
    width=width,
    color=[PASTEL[c] for c in PLOT_CATS],
    edgecolor="white",
    hatch="////",
    linewidth=0.6,
    zorder=3,
    label="Traditional model"
)

# Developed model bars (θ)
bars_dev = ax.bar(
    x + width/2,
    df_cmp[f"Developed θ={THETA:.2f} (B€)"].values,
    width=width,
    color=[PASTEL[c] for c in PLOT_CATS],
    edgecolor="white",
    linewidth=0.6,
    alpha=0.95,
    zorder=3,
    label="Developed model"
)

# Axes and style
ax.set_ylabel("System operator costs (billion €)")
ax.set_xticks(x)
ax.set_xticklabels(PLOT_CATS, rotation=18, ha="right")
ax.set_xlim(-0.6, len(PLOT_CATS)-0.4)
ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
ax.set_ylim(0, 3)

# Legend (only models)
handles = [
    mpatches.Patch(facecolor="#bea9a7", edgecolor="white", hatch="////", label="Traditional model"),
    mpatches.Patch(facecolor="#bea9a7", edgecolor="white", label="Developed model")
]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                ncol=2, frameon=True)
leg._legend_box.align = "left"
leg.get_frame().set_edgecolor("black")

plt.tight_layout()

outname = f"traditional_vs_developed_costs_plot_theta_{THETA:.2f}_{SCENARIO}.png"
outpath = Path("traditional_vs_developed_plots")
outpath.mkdir(exist_ok=True)
fig.savefig(outpath / outname, dpi=300, bbox_inches="tight")
print(f"✔ Saved figure: {outpath / outname}")

plt.show()
