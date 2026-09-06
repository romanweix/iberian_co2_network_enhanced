# plot_capex_detail.py
#
# Visualizes the CAPEX detail built by eus_full_analysis.build_capex_detail():
# onshore pipe CAPEX split into its insulation-only portion and the
# insulation-free remainder, and booster CAPEX kept split into its
# initial-build and additional (retrofit) components -- see the "CAPEX
# detail - supercritical" / "CAPEX detail - liquid" tabs in
# EUS_full_analysis.xlsx for the underlying tables.
#
# Two figures, one scenario axis each (supercritical scenarios on the
# left, liquid on the right, each group led by its benchmark run, as in
# plot_booster_vs_insulation.py):
#   capex_detail_absolute.png  -- two vertically-stacked panels, absolute
#       M€: (A) onshore pipe CAPEX, stacked into insulation-free + insulation;
#       (B) booster CAPEX, stacked into initial + additional. Two panels
#       rather than one dual-axis chart, for the same reason as
#       plot_booster_vs_insulation.py -- booster CAPEX is ~10x onshore-pipe
#       CAPEX, so a shared axis would flatten the onshore series.
#   capex_detail_relative.png -- one panel, % deviation from each scenario's
#       phase benchmark, grouped by scenario, for the four rows that have a
#       defined deviation (see note below on the insulation row).
#
# The "CAPEX onshore insulation" row is a memo-only breakout of the total,
# not one of the two summed here -- and its own % deviation is undefined
# for every scenario (both benchmark runs predate the insulation feature,
# so their insulation CAPEX is 0 -- see build_cost_category_vs_benchmark()'s
# divide-by-zero guard), so it is left out of the relative chart entirely
# rather than plotted as a misleading blank/zero bar.
#
# Usage (from the repo root):
#   python -m analysis.plot_capex_detail

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from analysis.eus_full_analysis import (
    SUPERCRITICAL_SCENARIOS, SUPERCRITICAL_BENCHMARK,
    LIQUID_SCENARIOS, LIQUID_BENCHMARK,
    CAPEX_DETAIL_ROWS,
    build_capex_detail, build_cost_category_vs_benchmark,
)

OUT_DIR = Path("analysis")

# Categorical palette slots 1/2/3/4 (blue/orange/aqua/yellow) from the
# validated default order in the dataviz skill's references/palette.md.
COLOR_ONSHORE_NO_INS = "#2a78d6"
COLOR_ONSHORE_INS = "#eb6834"
COLOR_BOOSTER_INITIAL = "#1baf7a"
COLOR_BOOSTER_ADDITIONAL = "#eda100"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#52514e"


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


def _group_labels(fig, ax, positions, supercritical_order, liquid_order):
    label_trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
    sc_mid = sum(positions[sk] for sk in supercritical_order) / len(supercritical_order)
    liq_mid = sum(positions[sk] for sk in liquid_order) / len(liquid_order)
    for mid, label in ((sc_mid, "Supercritical phase"), (liq_mid, "Liquid phase")):
        fig.text(mid, 0.085, label, transform=label_trans, ha="center", va="top",
                  fontsize=9.5, fontweight="bold", color=TEXT_MUTED)


def plot_capex_detail_absolute(capex_detail, out_dir: Path = OUT_DIR):
    supercritical_order = [SUPERCRITICAL_BENCHMARK] + SUPERCRITICAL_SCENARIOS
    liquid_order = [LIQUID_BENCHMARK] + LIQUID_SCENARIOS
    positions = _layout_positions(supercritical_order, liquid_order)
    all_scenarios = supercritical_order + liquid_order

    fig, (ax_onshore, ax_booster) = plt.subplots(
        2, 1, figsize=(11, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.07},
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.87, bottom=0.185)
    bar_width = 0.75

    for sk in all_scenarios:
        x = positions[sk]
        col = capex_detail[sk]
        ax_onshore.bar(x, col["CAPEX onshore pipeline without insulation"], bar_width,
                        color=COLOR_ONSHORE_NO_INS, edgecolor="white", linewidth=0.6, zorder=3)
        ax_onshore.bar(x, col["CAPEX onshore insulation"], bar_width,
                        bottom=col["CAPEX onshore pipeline without insulation"],
                        color=COLOR_ONSHORE_INS, edgecolor="white", linewidth=0.6, zorder=3)

        ax_booster.bar(x, col["CAPEX booster initial"], bar_width,
                        color=COLOR_BOOSTER_INITIAL, edgecolor="white", linewidth=0.6, zorder=3)
        ax_booster.bar(x, col["CAPEX booster additional"], bar_width,
                        bottom=col["CAPEX booster initial"],
                        color=COLOR_BOOSTER_ADDITIONAL, edgecolor="white", linewidth=0.6, zorder=3)

    ax_onshore.set_ylabel("Onshore pipeline CAPEX [M€]")
    ax_onshore.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_booster.set_ylabel("Booster CAPEX [M€]")
    ax_booster.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for ax in (ax_onshore, ax_booster):
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
    ax_onshore.spines["bottom"].set_visible(False)
    ax_booster.spines["bottom"].set_color(AXIS_COLOR)

    fig.suptitle("CAPEX detail by scenario (EUS): onshore pipeline vs. booster stations",
                 fontsize=13, fontweight="bold", y=0.975)

    legend_handles = [
        Patch(facecolor=COLOR_ONSHORE_NO_INS, label="Onshore pipeline (without insulation)"),
        Patch(facecolor=COLOR_ONSHORE_INS, label="Onshore insulation"),
        Patch(facecolor=COLOR_BOOSTER_INITIAL, label="Booster (initial build)"),
        Patch(facecolor=COLOR_BOOSTER_ADDITIONAL, label="Booster (additional/retrofit)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(legend_handles), frameon=False, fontsize=8.3)

    xs = [positions[sk] for sk in all_scenarios]
    ax_booster.set_xticks(xs)
    ax_booster.set_xticklabels([_scenario_label(sk) for sk in all_scenarios], fontsize=8.5)
    ax_booster.set_xlim(min(xs) - 1.0, max(xs) + 1.0)

    sep_x = (positions[supercritical_order[-1]] + positions[liquid_order[0]]) / 2
    for ax in (ax_onshore, ax_booster):
        ax.axvline(sep_x, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)

    _group_labels(fig, ax_booster, positions, supercritical_order, liquid_order)

    fig.text(
        0.01, 0.015,
        "* phase benchmark run (bm_sco2 / bm_dense_2), which predates the insulation feature "
        "(CAPEX onshore insulation = 0 there).\n"
        "Data: analysis/EUS_full_analysis.xlsx, 'CAPEX detail - supercritical'/'CAPEX detail - liquid' tabs (EUS).",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    png_path = out_dir / "capex_detail_absolute.png"
    pdf_path = out_dir / "capex_detail_absolute.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")


# Rows plotted in the relative-deviation chart, in legend/stacking order.
# "CAPEX onshore insulation" is excluded -- see the module docstring.
RELATIVE_ROWS = [
    ("CAPEX onshore pipeline (total, incl. insulation)", COLOR_ONSHORE_NO_INS),
    ("CAPEX onshore pipeline without insulation", COLOR_ONSHORE_INS),
    ("CAPEX booster initial", COLOR_BOOSTER_INITIAL),
    ("CAPEX booster additional", COLOR_BOOSTER_ADDITIONAL),
]


def plot_capex_detail_relative(capex_detail, out_dir: Path = OUT_DIR):
    sc_values, sc_pct = build_cost_category_vs_benchmark(
        capex_detail, SUPERCRITICAL_SCENARIOS, SUPERCRITICAL_BENCHMARK)
    liq_values, liq_pct = build_cost_category_vs_benchmark(
        capex_detail, LIQUID_SCENARIOS, LIQUID_BENCHMARK)

    pct_by_scenario = {}
    for sk in SUPERCRITICAL_SCENARIOS:
        pct_by_scenario[sk] = sc_pct[sk]
    for sk in LIQUID_SCENARIOS:
        pct_by_scenario[sk] = liq_pct[sk]
    # Benchmarks themselves are always exactly 0% deviation from themselves;
    # included on the axis (as a visual zero reference) but not fetched from
    # sc_pct/liq_pct, which only carry the sweep scenarios' own columns here.
    pct_by_scenario[SUPERCRITICAL_BENCHMARK] = {row: 0.0 for row, _ in RELATIVE_ROWS}
    pct_by_scenario[LIQUID_BENCHMARK] = {row: 0.0 for row, _ in RELATIVE_ROWS}

    supercritical_order = [SUPERCRITICAL_BENCHMARK] + SUPERCRITICAL_SCENARIOS
    liquid_order = [LIQUID_BENCHMARK] + LIQUID_SCENARIOS
    positions = _layout_positions(supercritical_order, liquid_order)
    all_scenarios = supercritical_order + liquid_order

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.2)

    n_series = len(RELATIVE_ROWS)
    group_width = 0.8
    bar_width = group_width / n_series

    for sk in all_scenarios:
        cx = positions[sk]
        for j, (row, color) in enumerate(RELATIVE_ROWS):
            v = pct_by_scenario[sk][row]
            x = cx - group_width / 2 + j * bar_width + bar_width / 2
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.bar(x, v, bar_width * 0.9, color=color, edgecolor="white", linewidth=0.4, zorder=3)

    ax.axhline(0, color=AXIS_COLOR, linewidth=1.2, zorder=2)
    ax.set_ylabel("Deviation from phase benchmark [%]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    fig.suptitle("CAPEX detail: % deviation from phase benchmark by scenario (EUS)",
                 fontsize=13, fontweight="bold", y=0.965)
    legend_handles = [Patch(facecolor=color, label=row) for row, color in RELATIVE_ROWS]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=2, frameon=False, fontsize=8.3)

    xs = [positions[sk] for sk in all_scenarios]
    ax.set_xticks(xs)
    ax.set_xticklabels([_scenario_label(sk) for sk in all_scenarios], fontsize=8.5)
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)

    sep_x = (positions[supercritical_order[-1]] + positions[liquid_order[0]]) / 2
    ax.axvline(sep_x, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)

    _group_labels(fig, ax, positions, supercritical_order, liquid_order)

    fig.text(
        0.01, 0.02,
        "* phase benchmark run (bm_sco2 / bm_dense_2) -- 0% by definition. "
        "'CAPEX onshore insulation' is omitted: its own deviation is undefined for every scenario "
        "(both benchmarks predate the insulation feature, so their insulation CAPEX is 0) -- see "
        "the 'CAPEX detail' tabs in EUS_full_analysis.xlsx for that row's absolute values instead.",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    png_path = out_dir / "capex_detail_relative.png"
    pdf_path = out_dir / "capex_detail_relative.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")


def main(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    capex_detail = build_capex_detail()
    plot_capex_detail_absolute(capex_detail, out_dir)
    plot_capex_detail_relative(capex_detail, out_dir)


if __name__ == "__main__":
    main()
