# plot_booster_cause_analysis.py
#
# Visualizes booster_cause_analysis.py's classification of why boosters
# were needed, straight from the checkpoint-level, two-pass analysis (0
# residual "Inconclusive" pipes here, unlike the older, single-pass,
# results_data-based classification in eus_full_analysis.py /
# plot_booster_vs_insulation.py, which leaves a few unresolved).
#
# Two standalone figures (one chart per file, for use one at a time in a
# thesis rather than as a combined multi-panel image):
#   booster_cause_by_scenario.png -- installed boosters per scenario,
#       stacked by Reason (Pressure only / Temperature only / Both), one
#       bar per insulation-surcharge sweep scenario, grouped by phase
#       (supercritical / liquid), each group led by its benchmark run (BM
#       predates the insulation/temperature feature, so only its total
#       booster count is available, drawn hatched -- see
#       build_booster_reasons()'s equivalent fallback in
#       eus_full_analysis.py).
#   booster_cause_by_phase.png -- reason share [%] by phase x insulation
#       status, showing how adding insulation shifts a scenario's boosters
#       away from temperature-driven and towards pressure-driven
#       (insulation removes the pipe's own heat loss as a constraint,
#       leaving pressure as the remaining one).
#
# Style: a classic print/journal look for thesis figures -- serif fonts
# (matching a LaTeX document's body text), thin black axes, minimal dotted
# gridlines, and a desaturated categorical palette (the muted
# blue/red/green triad common to matplotlib/seaborn's "deep"/"muted"
# themes) rather than the vivid Okabe-Ito set used previously. No in-figure
# title: the caption is expected to come from the thesis's own \caption{}.
#
# Usage (from the repo root):
#   python -m analysis.plot_booster_cause_analysis

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.booster_cause_analysis import (
    booster_cause_analysis, _checkpoint_files,
    REASON_ORDER, MODEL_RESULTS_DIR_DEFAULT, UTIL_W_DEFAULT,
)

OUT_DIR = Path("analysis")
RESULTS_DIR = Path("analysis/results_data")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.color": "#1a1a1a",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#c9c9c9",
    "grid.linewidth": 0.5,
    "grid.linestyle": ":",
    "legend.frameon": False,
})

# Desaturated categorical palette (matplotlib/seaborn "muted" triad) --
# print-friendly, low-saturation colors typical of journal figures.
COLOR_PRESSURE = "#4C72B0"      # muted blue
COLOR_TEMPERATURE = "#C44E52"   # muted brick red
COLOR_BOTH = "#55A868"          # muted green
COLOR_INCONCLUSIVE = "#8C8C8C"  # grey -- data-limitation label, not a physical category
COLOR_UNKNOWN_REASON = "#B0B0B0"  # BM runs: total known, reason split not
GRID_COLOR = "#c9c9c9"
AXIS_COLOR = "#333333"
TEXT_MUTED = "#404040"

REASON_COLORS = {
    "Pressure only": COLOR_PRESSURE,
    "Temperature only": COLOR_TEMPERATURE,
    "Both": COLOR_BOTH,
    "Inconclusive": COLOR_INCONCLUSIVE,
}

# Sweep scenario codes (see analysis/model_results/*_model_checkpoint.dill),
# each group led by its benchmark run. BM predates the insulation/temperature
# feature (excluded from booster_cause_analysis.py's own processing -- no
# per-pipe pressure/temperature/insulation data), so its total booster count
# is read separately, straight from the exported results_data workbook.
SUPERCRITICAL_ORDER = ["SCBM", "SCU", "SCI20", "SCI40", "SCI60", "SCI80", "SCI100", "SCI150"]
LIQUID_ORDER = ["LPBM", "LPU", "LPI20", "LPI40", "LPI60"]

SCENARIO_LABELS = {
    "SCBM": "SC-BM", "SCU": "SC-U", "SCI20": "SC-I20", "SCI40": "SC-I40",
    "SCI60": "SC-I60", "SCI80": "SC-I80", "SCI100": "SC-I100", "SCI150": "SC-I150",
    "LPBM": "LP-BM", "LPU": "LP-U", "LPI20": "LP-I20", "LPI40": "LP-I40", "LPI60": "LP-I60",
}

# BM total-booster source: (results_data filename stem, scenario code)
BM_SOURCES = {"SCBM": "bm_sco2", "LPBM": "bm_dense_2"}


def _bm_total_boosters(results_dir: Path) -> dict:
    totals = {}
    for code, stem in BM_SOURCES.items():
        f = results_dir / f"{stem}_stochastic_results_theta_1.00.xlsx"
        df = pd.read_excel(f, sheet_name="EUS - Pipes")
        installed = df[df["Installed"] == 1]
        totals[code] = installed["Number of boosters"].fillna(0).sum()
    return totals


def _layout_positions(supercritical_order, liquid_order):
    positions = {}
    x = 0.0
    for order in (supercritical_order, liquid_order):
        for i, sk in enumerate(order):
            positions[sk] = x
            x += 1.7 if i == 0 else 1.0
        x += 0.7
    return positions


def _style_axes(ax):
    """Classic journal-figure axes: a full box (all four spines), thin,
    with light dotted horizontal gridlines only -- rather than the
    open/minimalist (top+right spines removed) style used elsewhere in
    this repo's plot_*.py figures."""
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, linestyle=":", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8)


def _save(fig, name: str, out_dir: Path):
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector, for \includegraphics{} in LaTeX
    plt.close(fig)
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")


def _plot_by_scenario(boostcount: pd.DataFrame, bm_totals: dict, out_dir: Path):
    positions = _layout_positions(SUPERCRITICAL_ORDER, LIQUID_ORDER)
    all_scenarios = SUPERCRITICAL_ORDER + LIQUID_ORDER
    bm_codes = set(BM_SOURCES.keys())

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.2)
    bar_width = 0.75

    for sk in all_scenarios:
        x = positions[sk]
        if sk in bm_codes:
            ax.bar(x, bm_totals[sk], bar_width, color=COLOR_UNKNOWN_REASON,
                   edgecolor=TEXT_MUTED, linewidth=0.6, hatch="////", zorder=3)
            continue
        bottom = 0.0
        for reason in REASON_ORDER:
            v = boostcount.loc[sk, reason] if sk in boostcount.index else 0.0
            if v > 0:
                ax.bar(x, v, bar_width, bottom=bottom, color=REASON_COLORS[reason],
                       edgecolor="white", linewidth=0.6, zorder=3)
            bottom += v

    ax.set_ylabel("Installed boosters")
    _style_axes(ax)

    xs = [positions[sk] for sk in all_scenarios]
    ax.set_xticks(xs)
    ax.set_xticklabels([SCENARIO_LABELS[sk] for sk in all_scenarios], fontsize=8, rotation=45, ha="right")
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)

    sep_x = (positions[SUPERCRITICAL_ORDER[-1]] + positions[LIQUID_ORDER[0]]) / 2
    ax.axvline(sep_x, color=AXIS_COLOR, linewidth=0.6, linestyle=":", zorder=1)
    sc_mid = sum(positions[sk] for sk in SUPERCRITICAL_ORDER) / len(SUPERCRITICAL_ORDER)
    liq_mid = sum(positions[sk] for sk in LIQUID_ORDER) / len(LIQUID_ORDER)
    y_top = ax.get_ylim()[1]
    for mid, label in ((sc_mid, "Supercritical"), (liq_mid, "Liquid")):
        ax.text(mid, y_top * 1.03, label, ha="center", va="bottom",
                 fontsize=9.5, color=TEXT_MUTED)
    ax.set_ylim(top=y_top * 1.14)

    legend_handles = [Patch(facecolor=REASON_COLORS[r], edgecolor=AXIS_COLOR, linewidth=0.5, label=r)
                       for r in REASON_ORDER if boostcount[r].sum() > 0]
    legend_handles.append(Patch(facecolor=COLOR_UNKNOWN_REASON, edgecolor=AXIS_COLOR, linewidth=0.5,
                                 hatch="////", label="Total boosters (reason: pressure only)"))
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=8.5)

    fig.text(
        0.01, 0.01,
        "† pre-insulation-feature benchmark run (SC-BM / LP-BM): these predate the "
        "temperature constraint entirely, so pressure is the only possible cause; "
        "shown as a total rather than split, since they don't export the pipe-level "
        "columns the split is computed from.\n"
        "Data: analysis/model_results (base_utilization), two-pass classification -- "
        "see analysis/booster_cause_analysis.py / .xlsx.",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    _save(fig, "booster_cause_by_scenario", out_dir)


def _plot_by_phase(by_phase_ins: pd.DataFrame, out_dir: Path):
    groups = [
        ("supercritical", False, "Supercritical\n(no ins.)"),
        ("supercritical", True, "Supercritical\n(insulated)"),
        ("liquid/dense", False, "Liquid\n(no ins.)"),
    ]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.87, bottom=0.15)

    group_x = list(range(len(groups)))
    for gx, (phase, insulated, _) in zip(group_x, groups):
        if (phase, insulated) not in by_phase_ins.index:
            continue
        row = by_phase_ins.loc[(phase, insulated)]
        bottom = 0.0
        for reason in REASON_ORDER:
            v = row.get(f"{reason} [%]", 0.0)
            if v > 0:
                ax.bar(gx, v, 0.6, bottom=bottom, color=REASON_COLORS[reason],
                       edgecolor="white", linewidth=0.6, zorder=3)
            bottom += v

    ax.set_ylabel("Share of boosted pipes [%]")
    ax.set_ylim(0, 100)
    ax.set_xticks(group_x)
    ax.set_xticklabels([g[2] for g in groups], fontsize=9)
    _style_axes(ax)

    legend_handles = [Patch(facecolor=REASON_COLORS[r], edgecolor=AXIS_COLOR, linewidth=0.5, label=r)
                       for r in REASON_ORDER if (by_phase_ins[f"{r} [%]"] > 0).any()]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(legend_handles), frameon=False, fontsize=8.5)

    fig.text(
        0.01, 0.01,
        "Data: analysis/model_results (base_utilization), two-pass classification -- "
        "see analysis/booster_cause_analysis.py / .xlsx.",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    _save(fig, "booster_cause_by_phase", out_dir)


def plot_booster_cause_analysis(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _checkpoint_files(MODEL_RESULTS_DIR_DEFAULT)
    result = booster_cause_analysis(files, UTIL_W_DEFAULT)
    detail = result["detail"]
    by_phase_ins = result["by_phase_ins"].set_index(["Phase", "Insulated"])
    bm_totals = _bm_total_boosters(RESULTS_DIR)

    boostcount = (
        detail.groupby(["Scenario", "Reason"])["Number of boosters"].sum().unstack(fill_value=0.0)
    )
    for r in REASON_ORDER:
        if r not in boostcount.columns:
            boostcount[r] = 0.0

    _plot_by_scenario(boostcount, bm_totals, out_dir)
    _plot_by_phase(by_phase_ins, out_dir)


if __name__ == "__main__":
    plot_booster_cause_analysis()
