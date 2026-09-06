# plot_capex_detail.py
#
# Visualizes the CAPEX detail built by eus_full_analysis.build_capex_detail():
# onshore pipe CAPEX split into its insulation-only portion and the
# insulation-free remainder, and booster CAPEX kept split into its
# initial-build and additional (retrofit) components -- see the "CAPEX
# detail - supercritical" / "CAPEX detail - liquid" tabs in
# EUS_full_analysis.xlsx for the underlying tables.
#
# Three figures, one scenario axis each (supercritical scenarios on the
# left, liquid on the right). The benchmark runs (bm_sco2 / bm_dense_2,
# which predate the insulation feature) are never plotted here -- every
# reference below is instead each phase's own uninsulated sweep scenario,
# SC-U (noins) / LP-U (liq_noins), which -- unlike the benchmark runs --
# is a genuine point in the insulation-surcharge sweep:
#   capex_detail_absolute.png  -- two vertically-stacked panels, absolute
#       M€: (A) onshore pipe CAPEX, stacked into insulation-free + insulation;
#       (B) booster CAPEX, stacked into initial + additional. Two panels
#       rather than one dual-axis chart, for the same reason as
#       plot_booster_vs_insulation.py -- booster CAPEX is ~10x onshore-pipe
#       CAPEX, so a shared axis would flatten the onshore series.
#   capex_detail_relative.png -- one panel, % deviation from SC-U/LP-U,
#       grouped by scenario, for the four rows that have a defined
#       deviation (see note below on the insulation row).
#   capex_detail_trend.png -- 2x3 small multiples (one cell left empty),
#       one per CAPEX row, each plotting that row's value AS A LINE against
#       the insulation surcharge [%] -- an ordered quantity, unlike the
#       categorical scenario axis used in the other two figures -- with
#       supercritical and liquid as two separate lines so the two phases'
#       trends are directly comparable on shared axes. Four panels plot %
#       deviation from SC-U/LP-U (same four rows as capex_detail_relative.png);
#       the fifth ("CAPEX onshore insulation") plots the absolute M€ value
#       instead, since its deviation from SC-U/LP-U is undefined (see
#       INSULATION_ROW below).
#
# Plus one more figure per CAPEX_DETAIL_ROWS component (5 total,
# capex_component_*.png) -- the same trend line as one panel of
# capex_detail_trend.png, but each full-size and on its own, for use one
# at a time (e.g. in a thesis) rather than as a small multiple.
#
# The "CAPEX onshore insulation" row is a memo-only breakout of the total,
# not one of the two summed here -- and its own % deviation is undefined
# for every scenario (SC-U/LP-U have 0 insulation CAPEX by definition --
# no insulation -- see build_cost_category_vs_benchmark()'s divide-by-zero
# guard), so it is left out of the relative chart entirely rather than
# plotted as a misleading blank/zero bar.
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
    SUPERCRITICAL_SCENARIOS, LIQUID_SCENARIOS,
    CAPEX_DETAIL_ROWS,
    build_capex_detail, build_cost_category_vs_benchmark,
)

OUT_DIR = Path("analysis")

# Reference scenario per phase for every deviation computed in this module:
# each phase's own uninsulated sweep scenario, not the pre-insulation-feature
# benchmark run. SUPERCRITICAL_SCENARIOS / LIQUID_SCENARIOS both list their
# reference first (see eus_full_analysis.py), so the remaining entries are
# the insulated sweep compared against it.
SUPERCRITICAL_REFERENCE = SUPERCRITICAL_SCENARIOS[0]  # "noins"
LIQUID_REFERENCE = LIQUID_SCENARIOS[0]                # "liq_noins"
SUPERCRITICAL_SWEEP = SUPERCRITICAL_SCENARIOS[1:]     # ins20 .. ins150
LIQUID_SWEEP = LIQUID_SCENARIOS[1:]                   # liq_ins20 .. liq_ins60

# Categorical palette slots 1/2/3/4 (blue/orange/aqua/yellow) from the
# validated default order in the dataviz skill's references/palette.md.
COLOR_ONSHORE_NO_INS = "#2a78d6"
COLOR_ONSHORE_INS = "#eb6834"
COLOR_BOOSTER_INITIAL = "#1baf7a"
COLOR_BOOSTER_ADDITIONAL = "#eda100"
# Same two hues as plot_booster_vs_insulation.py's phase framing (and the
# earlier HTML dashboard this mirrors): orange = supercritical, blue =
# liquid. Reused here for the phase-trend chart's line series -- a
# different encoding than COLOR_ONSHORE_NO_INS/COLOR_ONSHORE_INS above
# (component, not phase), so kept as separate named constants.
COLOR_SUPERCRITICAL = "#eb6834"
COLOR_LIQUID = "#2a78d6"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#52514e"


# Official scenario abbreviations (thesis nomenclature): LP = liquid phase,
# SC = supercritical; -U = uninsulated (the reference scenario for every
# deviation in this module), -I<pct> = insulation surcharge. The benchmark
# runs (bm_sco2 / bm_dense_2) are never plotted here, so they have no entry.
SCENARIO_ABBR = {
    "liq_noins": "LP-U", "liq_ins20": "LP-I20", "liq_ins40": "LP-I40", "liq_ins60": "LP-I60",
    "noins": "SC-U", "ins20": "SC-I20", "ins40": "SC-I40",
    "ins60": "SC-I60", "ins80": "SC-I80", "ins100": "SC-I100", "ins150": "SC-I150",
}


def _scenario_label(scenario_key: str) -> str:
    return SCENARIO_ABBR.get(scenario_key, scenario_key)


def _layout_positions(supercritical_order, liquid_order):
    """x position per scenario: uniform spacing within each phase group,
    a wider gap between the two phase groups."""
    positions = {}
    x = 0.0
    for order in (supercritical_order, liquid_order):
        for sk in order:
            positions[sk] = x
            x += 1.0
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
    supercritical_order = SUPERCRITICAL_SCENARIOS
    liquid_order = LIQUID_SCENARIOS
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

        ax_booster.bar(x, col["CAPEX booster additional"], bar_width,
                        color=COLOR_BOOSTER_ADDITIONAL, edgecolor="white", linewidth=0.6, zorder=3)

    ax_onshore.set_ylabel("Onshore pipeline CAPEX [M€]")
    ax_onshore.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_booster.set_ylabel("Booster CAPEX\n(additional/retrofit only) [M€]")
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
        "SC-U/LP-U have 0 insulation CAPEX by definition (no insulation).\n"
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
        capex_detail, SUPERCRITICAL_SWEEP, SUPERCRITICAL_REFERENCE)
    liq_values, liq_pct = build_cost_category_vs_benchmark(
        capex_detail, LIQUID_SWEEP, LIQUID_REFERENCE)

    # sc_pct/liq_pct's columns are [reference] + sweep, i.e. exactly
    # SUPERCRITICAL_SCENARIOS / LIQUID_SCENARIOS -- the reference scenario's
    # own column is included and is exactly 0% (compared against itself),
    # so it needs no special-casing to appear as the chart's zero point.
    pct_by_scenario = {}
    for sk in SUPERCRITICAL_SCENARIOS:
        pct_by_scenario[sk] = sc_pct[sk]
    for sk in LIQUID_SCENARIOS:
        pct_by_scenario[sk] = liq_pct[sk]

    supercritical_order = SUPERCRITICAL_SCENARIOS
    liquid_order = LIQUID_SCENARIOS
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
    ax.set_ylabel("Deviation from SC-U / LP-U [%]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    fig.suptitle("CAPEX detail: % deviation from SC-U / LP-U by scenario (EUS)",
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
        "SC-U/LP-U -- 0% by definition (the reference). "
        "'CAPEX onshore insulation' is omitted: its own deviation is undefined for every scenario "
        "(SC-U/LP-U have 0 insulation CAPEX by definition) -- see "
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


def _surcharge_value(scenario_key: str) -> float:
    """Insulation surcharge [%] implied by a sweep scenario's key; "(liq_)noins"
    (SC-U / LP-U), this module's reference scenario, is the sweep's 0% point."""
    if scenario_key.endswith("noins"):
        return 0.0
    m = re.search(r"ins(\d+)$", scenario_key)
    if not m:
        raise ValueError(f"Not a sweep scenario: {scenario_key}")
    return float(m.group(1))


# "CAPEX onshore insulation" plotted alongside RELATIVE_ROWS' four %-deviation
# panels, but as its own absolute-M€ panel rather than a fifth % line: SC-U/
# LP-U have 0 insulation CAPEX by definition (no insulation), so this row's
# deviation from the reference is undefined for every scenario (see
# build_cost_category_vs_benchmark()'s divide-by-zero guard) -- there is no
# reference value to plot a trend of deviation from.
INSULATION_ROW = "CAPEX onshore insulation"


def plot_capex_detail_trend(capex_detail, out_dir: Path = OUT_DIR):
    sc_values, sc_pct = build_cost_category_vs_benchmark(
        capex_detail, SUPERCRITICAL_SWEEP, SUPERCRITICAL_REFERENCE)
    liq_values, liq_pct = build_cost_category_vs_benchmark(
        capex_detail, LIQUID_SWEEP, LIQUID_REFERENCE)

    sc_x = [_surcharge_value(sk) for sk in SUPERCRITICAL_SCENARIOS]
    liq_x = [_surcharge_value(sk) for sk in LIQUID_SCENARIOS]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.11, hspace=0.3, wspace=0.25)
    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    insulation_ax = axes[0, 2]
    axes[1, 2].set_visible(False)

    for ax, (row, _color) in zip(panel_axes, RELATIVE_ROWS):
        sc_y = [sc_pct[sk][row] for sk in SUPERCRITICAL_SCENARIOS]
        liq_y = [liq_pct[sk][row] for sk in LIQUID_SCENARIOS]

        ax.plot(sc_x, sc_y, marker="o", color=COLOR_SUPERCRITICAL, linewidth=2, markersize=5,
                label="Supercritical", zorder=3)
        ax.plot(liq_x, liq_y, marker="o", color=COLOR_LIQUID, linewidth=2, markersize=5,
                label="Liquid", zorder=3)

        ax.axhline(0, color=AXIS_COLOR, linewidth=1.0, zorder=2)
        ax.set_title(row, fontsize=9.5, color=TEXT_MUTED, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        ax.set_ylabel("Deviation from\nSC-U / LP-U [%]")

    # Insulation panel: absolute M€, not % deviation (see INSULATION_ROW note
    # above) -- same x-axis, its own y-axis/units, clearly labelled so it
    # isn't mistaken for a fifth deviation line.
    sc_ins_y = [capex_detail[sk][INSULATION_ROW] for sk in SUPERCRITICAL_SCENARIOS]
    liq_ins_y = [capex_detail[sk][INSULATION_ROW] for sk in LIQUID_SCENARIOS]
    insulation_ax.plot(sc_x, sc_ins_y, marker="o", color=COLOR_SUPERCRITICAL, linewidth=2,
                        markersize=5, label="Supercritical", zorder=3)
    insulation_ax.plot(liq_x, liq_ins_y, marker="o", color=COLOR_LIQUID, linewidth=2,
                        markersize=5, label="Liquid", zorder=3)
    insulation_ax.axhline(0, color=AXIS_COLOR, linewidth=1.0, zorder=2)
    insulation_ax.set_title(f"{INSULATION_ROW} (absolute, not vs. SC-U/LP-U†)",
                             fontsize=9.5, color=TEXT_MUTED, fontweight="bold")
    insulation_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    insulation_ax.set_ylabel("CAPEX [M€]")

    all_axes = panel_axes + [insulation_ax]
    for ax in all_axes:
        ax.grid(color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_color(AXIS_COLOR)
    for ax in (axes[1, 0], axes[1, 1], insulation_ax):
        ax.set_xlabel("Insulation surcharge [%]")
    all_x = sorted(set(sc_x) | set(liq_x))
    axes[0, 0].set_xticks(all_x)

    fig.suptitle("CAPEX detail: % deviation vs. insulation surcharge (EUS)",
                 fontsize=13, fontweight="bold", y=0.965)
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_SUPERCRITICAL, marker="o", linewidth=2, label="Supercritical"),
        plt.Line2D([0], [0], color=COLOR_LIQUID, marker="o", linewidth=2, label="Liquid"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=2, frameon=False, fontsize=9)

    fig.text(
        0.01, 0.02,
        "0% surcharge = SC-U / LP-U, this module's reference scenario (0% deviation by "
        "definition in the four left/center panels). Liquid sweep stops at +60%. "
        "† SC-U/LP-U have 0 insulation CAPEX by definition (no insulation), so this row's "
        "deviation from the reference is undefined -- its absolute value is shown instead.",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    png_path = out_dir / "capex_detail_trend.png"
    pdf_path = out_dir / "capex_detail_trend.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")


# (row, filename slug, line color) for every CAPEX_DETAIL_ROWS component,
# in the same order as CAPEX_DETAIL_ROWS. Filename slugs are independent
# of the row label text, so relabeling a row doesn't rename its file.
COMPONENT_SLUGS = [
    ("CAPEX onshore pipeline (total, incl. insulation)", "capex_component_onshore_total"),
    ("CAPEX onshore insulation", "capex_component_onshore_insulation"),
    ("CAPEX onshore pipeline without insulation", "capex_component_onshore_without_insulation"),
    ("CAPEX booster initial", "capex_component_booster_initial"),
    ("CAPEX booster additional", "capex_component_booster_additional"),
]


def plot_capex_detail_components(capex_detail, out_dir: Path = OUT_DIR):
    """One standalone figure per CAPEX_DETAIL_ROWS component (5 total,
    including "CAPEX onshore insulation" -- omitted from
    plot_capex_detail_relative()/the % panels of plot_capex_detail_trend()
    for the reason given at INSULATION_ROW, but included here on its own
    absolute-M€ axis like that trend chart's fifth panel)."""
    sc_values, sc_pct = build_cost_category_vs_benchmark(
        capex_detail, SUPERCRITICAL_SWEEP, SUPERCRITICAL_REFERENCE)
    liq_values, liq_pct = build_cost_category_vs_benchmark(
        capex_detail, LIQUID_SWEEP, LIQUID_REFERENCE)

    sc_x = [_surcharge_value(sk) for sk in SUPERCRITICAL_SCENARIOS]
    liq_x = [_surcharge_value(sk) for sk in LIQUID_SCENARIOS]
    all_x = sorted(set(sc_x) | set(liq_x))

    for row, slug in COMPONENT_SLUGS:
        is_insulation = row == INSULATION_ROW
        if is_insulation:
            sc_y = [capex_detail[sk][row] for sk in SUPERCRITICAL_SCENARIOS]
            liq_y = [capex_detail[sk][row] for sk in LIQUID_SCENARIOS]
        else:
            sc_y = [sc_pct[sk][row] for sk in SUPERCRITICAL_SCENARIOS]
            liq_y = [liq_pct[sk][row] for sk in LIQUID_SCENARIOS]

        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.subplots_adjust(left=0.13, right=0.97, top=0.85, bottom=0.24)

        ax.plot(sc_x, sc_y, marker="o", color=COLOR_SUPERCRITICAL, linewidth=2.2, markersize=6.5,
                label="Supercritical", zorder=3)
        ax.plot(liq_x, liq_y, marker="o", color=COLOR_LIQUID, linewidth=2.2, markersize=6.5,
                label="Liquid", zorder=3)
        ax.axhline(0, color=AXIS_COLOR, linewidth=1.0, zorder=2)

        if is_insulation:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.set_ylabel("CAPEX [M€]")
            title = f"{row} (absolute, not vs. SC-U/LP-U†)"
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
            ax.set_ylabel("Deviation from SC-U / LP-U [%]")
            title = row

        ax.set_xlabel("Insulation surcharge [%]")
        ax.set_xticks(all_x)
        ax.grid(color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_color(AXIS_COLOR)

        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.965)
        ax.legend(loc="best", frameon=False, fontsize=9.5)

        footnote = (
            "0% surcharge = SC-U / LP-U, this figure's reference scenario. Liquid sweep stops at +60%."
            if not is_insulation else
            "0% surcharge = SC-U / LP-U, this figure's reference scenario. Liquid sweep stops at +60%. "
            "† SC-U/LP-U have 0 insulation CAPEX by definition (no insulation), so this row's "
            "deviation from the reference is undefined -- its absolute value is shown instead."
        )
        fig.text(0.01, 0.02, footnote, fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom")

        png_path = out_dir / f"{slug}.png"
        pdf_path = out_dir / f"{slug}.pdf"
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
    plot_capex_detail_trend(capex_detail, out_dir)
    plot_capex_detail_components(capex_detail, out_dir)


if __name__ == "__main__":
    main()
