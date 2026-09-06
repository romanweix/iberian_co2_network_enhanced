# plot_booster_vs_insulation.py
#
# Visualizes the model's two competing ways of keeping a pipe above
# p_min/theta_min: build a booster station, or insulate the pipe so it
# needs boosting less. Two panels sharing one scenario axis (EUS,
# supercritical scenarios on the left, liquid on the right, each group
# led by its benchmark run):
#   (A) installed boosters per scenario, stacked by why they were needed
#       (pressure / temperature / both -- see build_booster_reasons() in
#       eus_full_analysis.py);
#   (B) the insulation shell volume [m^3] those scenarios' insulated
#       pipes represent.
# The two benchmark runs (bm_sco2, bm_dense_2) predate the insulation
# feature: panel A still shows their total booster count (just not the
# reason split, which needs pipe-level pressure/temperature columns they
# don't have), and panel B marks them "not modeled" rather than drawing a
# misleading zero.
#
# Deliberately two vertically-stacked panels on a shared x-axis rather
# than one dual-axis chart: booster count and insulation volume are
# different quantities on different scales, and overlaying them on twin
# y-axes invites a misread of which curve is "higher" at a glance.
#
# Usage (from the repo root):
#   python -m analysis.plot_booster_vs_insulation

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from analysis.compare_scenarios import compare_scenarios
from analysis.eus_full_analysis import (
    RESULTS_DIR, UTIL, REASON_ORDER,
    SUPERCRITICAL_SCENARIOS, SUPERCRITICAL_BENCHMARK,
    LIQUID_SCENARIOS, LIQUID_BENCHMARK,
    build_booster_reasons, build_insulation_volume,
)

OUT_DIR = Path("analysis")

# Categorical palette slots 1/2/3 (blue/orange/aqua) from the validated
# default order in the dataviz skill's references/palette.md -- assigned
# in fixed order to the three physical reasons; "Inconclusive" is muted
# ink rather than a 4th hue, since it is a data-limitation label, not a
# physical category on equal footing with the other three.
COLOR_PRESSURE = "#2a78d6"
COLOR_TEMPERATURE = "#eb6834"
COLOR_BOTH = "#1baf7a"
COLOR_INCONCLUSIVE = "#898781"
COLOR_UNKNOWN_REASON = "#c3c2b7"  # benchmark runs: total known, split not
COLOR_INSULATION = "#008300"      # categorical slot 6, unused by panel A
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#52514e"

REASON_COLORS = {
    "Pressure only": COLOR_PRESSURE,
    "Temperature only": COLOR_TEMPERATURE,
    "Both": COLOR_BOTH,
    "Inconclusive (installation-year snapshot)": COLOR_INCONCLUSIVE,
}


def _scenario_label(scenario_key: str) -> str:
    if scenario_key in (SUPERCRITICAL_BENCHMARK, LIQUID_BENCHMARK):
        return "BM*"
    if scenario_key.endswith("noins"):
        return "No ins."
    m = re.search(r"ins(\d+)$", scenario_key)
    return f"+{m.group(1)}%" if m else scenario_key


def _layout_positions(supercritical_order, liquid_order):
    """x position per scenario: a wider gap after each group's leading
    benchmark bar, and a wider gap still between the two phase groups."""
    positions = {}
    x = 0.0
    for order in (supercritical_order, liquid_order):
        for i, sk in enumerate(order):
            positions[sk] = x
            x += 1.7 if i == 0 else 1.0
        x += 0.7
    return positions


def plot_booster_vs_insulation(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compare_scenarios(RESULTS_DIR, UTIL)
    kpi_summary = result["kpi_summary"].set_index("Scenario")

    _, _, booster_boostcount, _ = build_booster_reasons()
    insulation_summary, _, _ = build_insulation_volume()
    insulation_by_scenario = insulation_summary.set_index("Scenario")

    supercritical_order = [SUPERCRITICAL_BENCHMARK] + SUPERCRITICAL_SCENARIOS
    liquid_order = [LIQUID_BENCHMARK] + LIQUID_SCENARIOS
    positions = _layout_positions(supercritical_order, liquid_order)
    all_scenarios = supercritical_order + liquid_order

    fig, (ax_boost, ax_ins) = plt.subplots(
        2, 1, figsize=(11, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.07},
    )
    # Fixed margins (not tight_layout): the phase-group labels and footnote
    # below the axes are placed in figure-fraction coordinates, so their
    # clearance has to be reserved explicitly rather than left to autolayout.
    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.185)
    bar_width = 0.75

    # --- Panel A: installed boosters, stacked by reason ------------------
    for sk in all_scenarios:
        x = positions[sk]
        if sk in booster_boostcount.columns:
            bottom = 0.0
            for reason in REASON_ORDER:
                v = booster_boostcount.loc[reason, sk]
                if v > 0:
                    ax_boost.bar(x, v, bar_width, bottom=bottom, color=REASON_COLORS[reason],
                                 edgecolor="white", linewidth=0.6, zorder=3)
                bottom += v
        else:
            total = kpi_summary.loc[sk, "Total boosters"]
            ax_boost.bar(x, total, bar_width, color=COLOR_UNKNOWN_REASON, edgecolor=TEXT_MUTED,
                         linewidth=0.6, hatch="////", zorder=3)

    ax_boost.set_ylabel("Installed boosters [count]")
    ax_boost.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_boost.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_boost.spines[spine].set_visible(False)
    ax_boost.spines["left"].set_color(AXIS_COLOR)
    ax_boost.spines["bottom"].set_visible(False)

    fig.suptitle("Booster requirement vs. insulation extent by scenario (EUS)",
                 fontsize=13, fontweight="bold", y=0.975)

    legend_handles = [Patch(facecolor=REASON_COLORS[r], label=r.replace(
        " (installation-year snapshot)", "*")) for r in REASON_ORDER]
    legend_handles.append(Patch(facecolor=COLOR_UNKNOWN_REASON, edgecolor=TEXT_MUTED, hatch="////",
                                 label="Total boosters (reason not available)†"))
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(legend_handles), frameon=False, fontsize=8.3)

    # --- Panel B: insulation shell volume ---------------------------------
    volumes = []
    for sk in all_scenarios:
        x = positions[sk]
        row = insulation_by_scenario.loc[sk]
        vol = row["Insulation volume [m³]"]
        if vol is None or (isinstance(vol, float) and np.isnan(vol)):
            ax_ins.text(x, 0, "n/a‡", ha="center", va="bottom", fontsize=7.5,
                        color=TEXT_MUTED, style="italic")
        else:
            volumes.append(vol)
            ax_ins.bar(x, vol, bar_width, color=COLOR_INSULATION, edgecolor="white",
                       linewidth=0.6, zorder=3)

    ax_ins.set_ylabel("Insulation shell volume [m³]")
    ax_ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_ins.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_ins.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_ins.spines[spine].set_visible(False)
    ax_ins.spines["left"].set_color(AXIS_COLOR)
    ax_ins.spines["bottom"].set_color(AXIS_COLOR)

    # --- Shared x-axis: scenario labels, group separator, group titles ---
    xs = [positions[sk] for sk in all_scenarios]
    ax_ins.set_xticks(xs)
    ax_ins.set_xticklabels([_scenario_label(sk) for sk in all_scenarios], fontsize=8.5)
    ax_ins.set_xlim(min(xs) - 1.0, max(xs) + 1.0)

    sep_x = (positions[supercritical_order[-1]] + positions[liquid_order[0]]) / 2
    for ax in (ax_boost, ax_ins):
        ax.axvline(sep_x, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)

    # Figure-fraction y for the group labels / footnote, keyed to data-space
    # x on ax_ins -- stable under the fixed subplots_adjust margins above,
    # unlike axes-fraction text which would shift if the layout changes.
    label_trans = transforms.blended_transform_factory(ax_ins.transData, fig.transFigure)
    sc_mid = sum(positions[sk] for sk in supercritical_order) / len(supercritical_order)
    liq_mid = sum(positions[sk] for sk in liquid_order) / len(liquid_order)
    for mid, label in ((sc_mid, "Supercritical phase"), (liq_mid, "Liquid phase")):
        fig.text(mid, 0.085, label, transform=label_trans, ha="center", va="top",
                  fontsize=9.5, fontweight="bold", color=TEXT_MUTED)

    fig.text(
        0.01, 0.015,
        "* pre-insulation-feature benchmark run (bm_sco2 / bm_dense_2).  "
        "† benchmark total boosters shown; pressure/temperature split needs pipe-level columns "
        "these runs don't export.  ‡ insulation not modeled in this run.\n"
        "Insulation volume = π·((r+0.15 m)²−r²)·length per insulated pipe (15 cm shell, nominal "
        "pipe diameter as outer radius).  Data: analysis/results_data (EUS).",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    png_path = out_dir / "booster_vs_insulation.png"
    pdf_path = out_dir / "booster_vs_insulation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector, for \includegraphics{} in LaTeX
    plt.close(fig)

    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")


if __name__ == "__main__":
    plot_booster_vs_insulation()
