# booster_cause_analysis.py
#
# For every installed pipe that has a booster station in the model results,
# classifies WHY the booster was needed: because the origin pressure alone
# would let the critical (highest-elevation) or lowest-pressure point fall
# below p_min ("Pressure"), because the origin temperature alone would let
# the critical-pressure-point or coldest point fall below theta_min
# ("Temperature"), or both ("Both"). Mirrors analysis/compare_scenarios.py:
# same source data (the "{util} - Pipes" sheet exported by
# developed_solution.py), same workbook-per-scenario handling.
#
# By default scans every *_stochastic_results_theta_*.xlsx in
# analysis/results_data (all insulation-surcharge / phase-state scenarios).
# Pass --file to analyze a single workbook instead, e.g. the latest model
# run in output_developed_model/.
#
# Usage (from the repo root):
#   python -m analysis.booster_cause_analysis
#   python -m analysis.booster_cause_analysis --file output_developed_model/stochastic_results_theta_1.00.xlsx
#   python -m analysis.booster_cause_analysis --results-dir analysis/results_data --util EUS

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.compare_scenarios import _parse_scenario_label

# Defaults match the current model run (see iberian_co2_network/data.py:
# THETA_MIN = 32.0 for the CO2 phase used in these results) and
# developed_model.py's "p_min": 100.0 default.
P_MIN_DEFAULT = 100.0
THETA_MIN_DEFAULT = 32.0

# Absolute tolerance (bar / °C) below which a margin is treated as exactly
# zero rather than a floating-point artifact of the LP solution.
TOL = 1e-6

CAUSE_COLS = [
    "Pipe ID",
    "Number of boosters",
    "Booster installation year",
    "Pressure at highest point [bar]",
    "Lowest pressure [bar]",
    "Pressure margin (no booster) [bar]",
    "Temperature at critical pressure point [°C]",
    "Lowest temperature [°C]",
    "Temperature margin (no booster) [°C]",
    "Cause",
]


def _classify_causes(df: pd.DataFrame, p_min: float, theta_min: float) -> pd.DataFrame:
    """Return one row per installed, boosted pipe with its pressure/temperature
    margin (without the booster's contribution) and a resulting Cause label.

    A negative margin means that point would fall below the p_min/theta_min
    threshold without the booster -- i.e. the booster is required to satisfy
    that constraint. Both margins are always computed; a pipe can (and often
    does) need its booster for both reasons at once, since the same distance
    that erodes pressure also erodes temperature.
    """
    boosted = df[(df["Installed"] == 1) & (df["Number of boosters"].fillna(0) > 0)].copy()

    pressure_margin = pd.concat(
        [boosted["Pressure at highest point [bar]"], boosted["Lowest pressure [bar]"]], axis=1
    ).min(axis=1) - p_min

    # Older benchmark runs (e.g. bm_sco2/bm_dense_2) predate the temperature
    # feature and export no temperature columns at all -- every booster there
    # is by construction pressure-only, so give it a temperature margin of
    # +inf (never binding) instead of failing on the missing columns.
    has_temp_cols = "Temperature at critical pressure point [°C]" in df.columns
    if has_temp_cols:
        temperature_margin = pd.concat(
            [boosted["Temperature at critical pressure point [°C]"], boosted["Lowest temperature [°C]"]], axis=1
        ).min(axis=1) - theta_min
    else:
        temperature_margin = pd.Series(float("inf"), index=boosted.index)

    boosted["Pressure margin (no booster) [bar]"] = pressure_margin
    boosted["Temperature margin (no booster) [°C]"] = temperature_margin if has_temp_cols else None

    pressure_binds = pressure_margin < -TOL
    temperature_binds = temperature_margin < -TOL

    def _cause(p_bind: bool, t_bind: bool) -> str:
        if p_bind and t_bind:
            return "Both"
        if p_bind:
            return "Pressure"
        if t_bind:
            return "Temperature"
        return "Unclear"  # booster present but neither margin is violated (e.g. Big-M slack)

    boosted["Cause"] = [
        _cause(p_bind, t_bind) for p_bind, t_bind in zip(pressure_binds, temperature_binds)
    ]

    # Older benchmark runs are also missing "Booster installation year" and
    # the raw temperature columns entirely (not just the margin computed
    # above) -- add them as NaN so every scenario's detail rows line up under
    # the same columns.
    for col in CAUSE_COLS:
        if col not in boosted.columns:
            boosted[col] = None

    return boosted[CAUSE_COLS]


def _load_boosted_pipes(xlsx_path: Path, util: str, p_min: float, theta_min: float) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=f"{util} - Pipes")
    return _classify_causes(df, p_min, theta_min)


def booster_cause_analysis(
    files: list[Path], util: str, p_min: float, theta_min: float
) -> dict[str, pd.DataFrame]:
    detail_frames = []
    summary_rows = []

    for f in files:
        scenario_key, phase, surcharge_pct = _parse_scenario_label(f.name)
        boosted = _load_boosted_pipes(f, util, p_min, theta_min)

        detail = boosted.copy()
        detail.insert(0, "Scenario", scenario_key)
        detail_frames.append(detail)

        counts = boosted["Cause"].value_counts()
        summary_rows.append({
            "Scenario": scenario_key,
            "Phase": phase,
            "Insulation surcharge [%]": surcharge_pct,
            "Total boosters": len(boosted),
            "Pressure-caused": int(counts.get("Pressure", 0)),
            "Temperature-caused": int(counts.get("Temperature", 0)),
            "Both": int(counts.get("Both", 0)),
            "Unclear": int(counts.get("Unclear", 0)),
        })

    booster_detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    cause_summary = pd.DataFrame(summary_rows).sort_values(
        ["Phase", "Insulation surcharge [%]"], na_position="first"
    )

    return {"booster_detail": booster_detail, "cause_summary": cause_summary}


def _plot_cause_summary(cause_summary: pd.DataFrame, out_path: Path):
    cause_cols = ["Pressure-caused", "Temperature-caused", "Both", "Unclear"]
    cause_cols = [c for c in cause_cols if cause_summary[c].sum() > 0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(cause_summary))
    bottom = [0.0] * len(cause_summary)
    for cause in cause_cols:
        values = cause_summary[cause].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=cause)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(cause_summary["Scenario"], rotation=45, ha="right")
    ax.set_ylabel("Number of boosters")
    ax.set_title("Booster installation cause by scenario")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("analysis/results_data"),
                         help="Directory of *_stochastic_results_theta_*.xlsx workbooks (ignored if --file is given).")
    parser.add_argument("--file", type=Path, default=None,
                         help="Analyze a single model-results workbook instead of scanning --results-dir, "
                              "e.g. output_developed_model/stochastic_results_theta_1.00.xlsx")
    parser.add_argument("--util", default="EUS", help="Utilization-scenario sheet prefix (LUS/EUS/HUS)")
    parser.add_argument("--p-min", type=float, default=P_MIN_DEFAULT, help="Minimum allowed pressure [bar]")
    parser.add_argument("--theta-min", type=float, default=THETA_MIN_DEFAULT, help="Minimum allowed temperature [°C]")
    parser.add_argument("--out", type=Path, default=Path("analysis/booster_cause_analysis.xlsx"))
    args = parser.parse_args()

    if args.file is not None:
        files = [args.file]
    else:
        files = sorted(args.results_dir.glob("*_stochastic_results_theta_*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No *_stochastic_results_theta_*.xlsx files found in {args.results_dir}")

    result = booster_cause_analysis(files, args.util, args.p_min, args.theta_min)
    booster_detail, cause_summary = result["booster_detail"], result["cause_summary"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        cause_summary.to_excel(writer, sheet_name="Cause summary", index=False)
        booster_detail.to_excel(writer, sheet_name="Booster detail", index=False)

    print(f"✔  Cause analysis workbook written to {args.out.resolve()}")

    if len(cause_summary) > 1:
        plots_dir = args.out.parent / "scenario_comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "booster_cause_by_scenario.png"
        _plot_cause_summary(cause_summary, plot_path)
        print(f"✔  Plot written to {plot_path.resolve()}")

    print("\n--- Cause summary ---")
    print(cause_summary.to_string(index=False))


if __name__ == "__main__":
    main()
