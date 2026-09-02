# compare_scenarios.py
#
# Systematic comparison across the insulation-surcharge / phase-state
# scenario runs in analysis/results_data/*.xlsx, for the master's thesis
# evaluation. Reads every stochastic_results_*.xlsx in the results folder,
# extracts the cost breakdown and pipe-level KPIs for one utilization
# scenario (default: EUS = base_utilization), and writes a single
# comparison workbook plus a few overview charts.
#
# Usage (from the repo root):
#   python -m analysis.compare_scenarios
#   python -m analysis.compare_scenarios --results-dir analysis/results_data --util EUS

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rows in the cost breakdown that are informational-only breakdowns of
# another row's value (see developed_solution.py's `memo_only` set) and
# must be excluded when summing to a Gross/Net total, to avoid double
# counting.
MEMO_ONLY_ROWS = {
    "CAPEX onshore pipe (insulation only)",
    "OPEX onshore pipe (insulation only)",
}

# Rows the thesis's own "Gross cost" definition excludes: capture cost and
# the uncaptured-emissions penalty are tracked separately, not counted as
# part of the transport-network cost being compared across scenarios.
GROSS_COST_EXCLUDED_ROWS = {
    "Capture cost",
    "Penalty for uncaptured emissions",
}

# The "Total" column in each Cost breakdown sheet is in M€ (developed_solution.py).
# The thesis's comparison table presents Gross/Net cost in Bn€, so divide by
# 1000 here too -- verified against the already hand-built
# comparison_results_EUS_costs.xlsx (Gross/Net cost = 45.209 / 14.852 Bn€
# for "noins").
M_EUR_TO_BN_EUR_SCALE = 1000.0

NOT_INSULATED_U = 2.0


def _parse_scenario_label(filename: str) -> tuple[str, str, float | None]:
    """Derive (scenario_key, phase, surcharge_pct) from a results filename.

    Examples:
      noins_stochastic_results_theta_1.00.xlsx        -> ("noins", "supercritical", None)
      ins60_stochastic_results_theta_1.00.xlsx         -> ("ins60", "supercritical", 60.0)
      liq_ins40_stochastic_results_theta_1.00.xlsx     -> ("liq_ins40", "liquid", 40.0)
      liq_noins_stochastic_results_theta_1.00.xlsx     -> ("liq_noins", "liquid", None)
    """
    stem = filename.split("_stochastic_results")[0]
    phase = "liquid" if stem.startswith("liq_") else "supercritical"
    tag = stem[len("liq_"):] if phase == "liquid" else stem

    m = re.match(r"ins(\d+)$", tag)
    surcharge_pct = float(m.group(1)) if m else None
    return stem, phase, surcharge_pct


def _load_cost_breakdown(xlsx_path: Path, util: str) -> pd.Series:
    """Return the 'Total' column of the {util} - Cost breakdown sheet, indexed by Concept."""
    df = pd.read_excel(xlsx_path, sheet_name=f"{util} - Cost breakdown")
    df = df[df["Concept"] != "TOTAL"]
    return df.set_index("Concept")["Total"]


def _gross_net_cost(cost_series: pd.Series) -> tuple[float, float]:
    """Gross cost = sum of all cost rows except memo rows, CO2 sales revenue,
    capture cost, and the uncaptured-emissions penalty (tracked separately).
    Net cost = Gross cost + CO2 sales revenue (revenue is stored as a
    negative value). Both are converted from M€ to Bn€
    (see M_EUR_TO_BN_EUR_SCALE)."""
    excluded = MEMO_ONLY_ROWS | GROSS_COST_EXCLUDED_ROWS | {"CO2 sales revenue"}
    real_rows = cost_series.drop(index=[c for c in excluded if c in cost_series.index])
    revenue = cost_series.get("CO2 sales revenue", 0.0)
    gross = real_rows.sum() / M_EUR_TO_BN_EUR_SCALE
    net = gross + revenue / M_EUR_TO_BN_EUR_SCALE
    return gross, net


def _load_pipe_kpis(xlsx_path: Path, util: str) -> dict:
    """Aggregate pipe-level KPIs for one scenario: installed count/length,
    insulation share (by length), total boosters, total pipe PV cost, plus
    a per-diameter breakdown of how many pipelines and how many booster
    stations (summed "Number of boosters") were built at each diameter."""
    df = pd.read_excel(xlsx_path, sheet_name=f"{util} - Pipes")
    # A few older/benchmark runs (bm_sco2, bm_dense_2) predate the
    # insulation feature and don't carry these columns at all -- treat
    # every pipe there as uninsulated rather than failing.
    if "Insulated" not in df.columns:
        df["Insulated"] = False
    if "Insulation cost [M€]" not in df.columns:
        df["Insulation cost [M€]"] = 0.0
    installed = df[df["Installed"] == 1]

    total_length = installed["Longitude [km]"].sum()
    insulated_length = installed.loc[installed["Insulated"] == True, "Longitude [km]"].sum()  # noqa: E712
    insulated_share_pct = 100.0 * insulated_length / total_length if total_length > 0 else 0.0

    kpis = {
        "Installed pipes": len(installed),
        "Total pipe length [km]": total_length,
        "Insulated share (by length) [%]": insulated_share_pct,
        "Total boosters": installed["Number of boosters"].fillna(0).sum(),
        "Avg. diameter [inch]": installed["Diameter [inch]"].mean() if len(installed) else None,
        "Total pipe PV cost [M€]": installed["Present Value Cost [M€]"].sum(),
        "Total insulation CAPEX [M€]": installed["Insulation cost [M€]"].sum(),
    }

    # Per-diameter breakdown. Diameter [inch] is exported as an int/float,
    # so sort numerically for a sensible column order (Pipes D=6", D=10", ...).
    by_diam = installed.groupby("Diameter [inch]")
    for diam, sub in sorted(by_diam, key=lambda kv: kv[0]):
        diam_label = f'{int(diam)}"'
        insulated_sub = sub[sub["Insulated"] == True]  # noqa: E712
        kpis[f"Pipes D={diam_label}"] = len(sub)
        kpis[f"Boosters D={diam_label}"] = sub["Number of boosters"].fillna(0).sum()
        kpis[f"Insulated pipes D={diam_label}"] = len(insulated_sub)
        kpis[f"Insulated length D={diam_label} [km]"] = insulated_sub["Longitude [km]"].sum()

    return kpis


def compare_scenarios(results_dir: Path, util: str = "EUS") -> dict[str, pd.DataFrame]:
    files = sorted(results_dir.glob("*_stochastic_results_theta_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No *_stochastic_results_theta_*.xlsx files found in {results_dir}")

    cost_cols = {}
    kpi_rows = []

    for f in files:
        scenario_key, phase, surcharge_pct = _parse_scenario_label(f.name)
        cost_series = _load_cost_breakdown(f, util)
        cost_cols[scenario_key] = cost_series

        gross, net = _gross_net_cost(cost_series)
        pipe_kpis = _load_pipe_kpis(f, util)

        kpi_rows.append({
            "Scenario": scenario_key,
            "Phase": phase,
            "Insulation surcharge [%]": surcharge_pct,
            "Gross cost [Bn€]": gross,
            "Net cost [Bn€]": net,
            **pipe_kpis,
        })

    cost_comparison = pd.DataFrame(cost_cols)
    cost_comparison.index.name = "Concept"

    kpi_summary = pd.DataFrame(kpi_rows).sort_values(["Phase", "Insulation surcharge [%]"], na_position="first")

    # Not every scenario builds at every diameter -- a diameter unused in one
    # scenario but used in another leaves NaN there after pd.DataFrame(kpi_rows)
    # aligns columns; these are counts/lengths, so 0 is the correct fill,
    # not NaN (which would misleadingly suggest "unknown" rather than "none").
    per_diam_prefixes = ("Pipes D=", "Boosters D=", "Insulated pipes D=", "Insulated length D=")
    per_diam_cols = [c for c in kpi_summary.columns if c.startswith(per_diam_prefixes)]
    kpi_summary[per_diam_cols] = kpi_summary[per_diam_cols].fillna(0)

    # Columns get appended in whatever order each scenario's diameters were
    # first seen, so re-sort the per-diameter block numerically by diameter,
    # grouping the 4 metrics for each diameter together.
    def _diam_sort_key(col):
        m = re.search(r'D=(\d+)"', col)
        diam = int(m.group(1))
        metric_rank = per_diam_prefixes.index(next(p for p in per_diam_prefixes if col.startswith(p)))
        return (diam, metric_rank)

    other_cols = [c for c in kpi_summary.columns if c not in per_diam_cols]
    kpi_summary = kpi_summary[other_cols + sorted(per_diam_cols, key=_diam_sort_key)]

    return {"cost_comparison": cost_comparison, "kpi_summary": kpi_summary}


def _plot_kpi(kpi_summary: pd.DataFrame, column: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for phase, sub in kpi_summary.groupby("Phase"):
        sub = sub.dropna(subset=["Insulation surcharge [%]"]).sort_values("Insulation surcharge [%]")
        if sub.empty:
            continue
        ax.plot(sub["Insulation surcharge [%]"], sub[column], marker="o", label=phase)

        # Also show the "noins" baseline (surcharge = None) as a horizontal reference line
        noins = kpi_summary[(kpi_summary["Phase"] == phase) & (kpi_summary["Insulation surcharge [%]"].isna())]
        if not noins.empty:
            ax.axhline(noins[column].iloc[0], linestyle="--", alpha=0.5,
                       label=f"{phase} (no insulation)")

    ax.set_xlabel("Insulation surcharge [%]")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs. insulation surcharge")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("analysis/results_data"))
    parser.add_argument("--util", default="EUS", help="Utilization-scenario sheet prefix (LUS/EUS/HUS)")
    parser.add_argument("--out", type=Path, default=Path("analysis/scenario_comparison.xlsx"))
    args = parser.parse_args()

    result = compare_scenarios(args.results_dir, args.util)
    cost_comparison, kpi_summary = result["cost_comparison"], result["kpi_summary"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        cost_comparison.to_excel(writer, sheet_name="Cost breakdown comparison")
        kpi_summary.to_excel(writer, sheet_name="KPI summary", index=False)

    print(f"✔  Comparison workbook written to {args.out.resolve()}")

    plots_dir = args.out.parent / "scenario_comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_kpi(kpi_summary, "Net cost [Bn€]", "Net cost [Bn€]", plots_dir / "net_cost_vs_surcharge.png")
    _plot_kpi(kpi_summary, "Insulated share (by length) [%]", "Insulated share (by length) [%]",
              plots_dir / "insulated_share_vs_surcharge.png")
    _plot_kpi(kpi_summary, "Total boosters", "Total boosters (installed)",
              plots_dir / "total_boosters_vs_surcharge.png")
    print(f"✔  Plots written to {plots_dir.resolve()}")

    print("\n--- KPI summary ---")
    print(kpi_summary.to_string(index=False))


if __name__ == "__main__":
    main()
