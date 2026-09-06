# plot_booster_insulation_by_diameter.py
#
# Visualizes, for every scenario, how installed booster stations and
# insulated pipe segments distribute across pipeline diameter -- laid out
# like plot_booster_vs_insulation.py (one scenario axis, supercritical
# scenarios on the left, liquid on the right, each group led by its
# benchmark run), but with each bar now stacked by diameter class instead
# of a single total:
#   (A) number of installed boosters per scenario, stacked by the
#       diameter of the pipe they sit on;
#   (B) number of insulated pipe segments per scenario, stacked the same
#       way.
# The two benchmark runs (bm_sco2, bm_dense_2) predate the insulation
# feature and have no "Insulated" column at all -- panel B marks them
# "not modeled" rather than drawing a misleading empty bar (panel A does
# show them: boosters and diameter are tracked in both schema versions).
#
# Diameter is an ORDERED quantity, not a set of unrelated categories, so
# both panels use the dataviz skill's single-hue sequential/ordinal blue
# ramp (light = small diameter, dark = large) rather than distinct
# categorical hues -- the palette's 8-slot categorical set isn't built to
# stay distinguishable across the 10 diameter classes in this network
# anyway.
#
# Usage (from the repo root):
#   python -m analysis.plot_booster_insulation_by_diameter

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from analysis.eus_full_analysis import (
    UTIL,
    SUPERCRITICAL_SCENARIOS, SUPERCRITICAL_BENCHMARK,
    LIQUID_SCENARIOS, LIQUID_BENCHMARK,
    labeled_scenarios,
)

OUT_DIR = Path("analysis")

# Sequential blue ramp, steps 250->700 from the dataviz skill's
# references/palette.md (light->dark; step 250 is the ordinal floor for
# the light chart surface used here -- see the palette's "ordinal ramp"
# note). Ten steps for the ten diameter classes actually built in this
# network (6" to 42" in 4" increments); assigned once diameters are known
# at runtime rather than hard-coded, so it stays correct if the candidate
# diameter set ever changes.
SEQUENTIAL_BLUE_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
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


def build_booster_and_insulation_by_diameter():
    """(booster_by_diam, insulated_by_diam, diameters, excluded_insulation):
    booster_by_diam / insulated_by_diam map scenario -> {diameter_inch: count}
    (installed pipes only); insulated_by_diam is None for a scenario whose
    Pipes sheet has no "Insulated" column (the two benchmark runs).
    diameters is the sorted union of every diameter actually built,
    across every scenario."""
    booster_by_diam, insulated_by_diam = {}, {}
    excluded_insulation = []
    all_diameters = set()

    for f, scenario_key, phase_label, surcharge_pct in labeled_scenarios():
        df = pd.read_excel(f, sheet_name=f"{UTIL} - Pipes")
        installed = df[df["Installed"] == 1]
        all_diameters.update(int(d) for d in installed["Diameter [inch]"].dropna().unique())

        boosters = installed.groupby("Diameter [inch]")["Number of boosters"].sum()
        booster_by_diam[scenario_key] = {int(d): v for d, v in boosters.items()}

        if "Insulated" not in df.columns:
            excluded_insulation.append(scenario_key)
            insulated_by_diam[scenario_key] = None
        else:
            insulated = installed[installed["Insulated"] == True]  # noqa: E712
            counts = insulated.groupby("Diameter [inch]").size()
            insulated_by_diam[scenario_key] = {int(d): v for d, v in counts.items()}

    return booster_by_diam, insulated_by_diam, sorted(all_diameters), excluded_insulation


def plot_booster_insulation_by_diameter(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    booster_by_diam, insulated_by_diam, diameters, excluded_insulation = \
        build_booster_and_insulation_by_diameter()

    if len(diameters) > len(SEQUENTIAL_BLUE_STEPS):
        raise ValueError(f"{len(diameters)} diameter classes found but only "
                          f"{len(SEQUENTIAL_BLUE_STEPS)} ramp steps defined; add more steps.")
    # Evenly re-sample the 10-step ramp if fewer than 10 diameters are
    # actually present, keeping the lightest/darkest anchors.
    idx = np.linspace(0, len(SEQUENTIAL_BLUE_STEPS) - 1, len(diameters)).round().astype(int)
    diam_colors = {d: SEQUENTIAL_BLUE_STEPS[i] for d, i in zip(diameters, idx)}

    supercritical_order = [SUPERCRITICAL_BENCHMARK] + SUPERCRITICAL_SCENARIOS
    liquid_order = [LIQUID_BENCHMARK] + LIQUID_SCENARIOS
    positions = _layout_positions(supercritical_order, liquid_order)
    all_scenarios = supercritical_order + liquid_order
    bar_width = 0.75

    fig, (ax_boost, ax_ins) = plt.subplots(
        2, 1, figsize=(11, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )
    # Fixed margins (not tight_layout): the phase-group labels and footnote
    # below the axes are placed in figure-fraction coordinates, so their
    # clearance has to be reserved explicitly rather than left to autolayout.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.185)

    # --- Panel A: installed boosters, stacked by diameter -----------------
    for sk in all_scenarios:
        x = positions[sk]
        bottom = 0.0
        for d in diameters:
            v = booster_by_diam[sk].get(d, 0)
            if v > 0:
                ax_boost.bar(x, v, bar_width, bottom=bottom, color=diam_colors[d],
                             edgecolor="white", linewidth=0.6, zorder=3)
            bottom += v

    ax_boost.set_ylabel("Installed boosters [count]")
    ax_boost.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_boost.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_boost.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_boost.spines[spine].set_visible(False)
    ax_boost.spines["left"].set_color(AXIS_COLOR)
    ax_boost.spines["bottom"].set_visible(False)
    ax_boost.tick_params(axis="x", which="both", bottom=False)  # sharex hides labels, not tick marks
    ax_boost.set_title("Installed boosters, by pipe diameter", fontsize=10.5, fontweight="bold",
                        color=TEXT_MUTED, loc="left", pad=4)

    fig.suptitle("Booster and insulation distribution across pipe diameter, by scenario (EUS)",
                 fontsize=12.5, fontweight="bold", y=0.975)

    legend_handles = [Patch(facecolor=diam_colors[d], label=f'{d}"') for d in diameters]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), title="Diameter",
               ncol=len(diameters), frameon=False, fontsize=8.3, title_fontsize=8.3)

    # --- Panel B: insulated pipe segments, stacked by diameter -------------
    for sk in all_scenarios:
        x = positions[sk]
        counts = insulated_by_diam[sk]
        if counts is None:
            ax_ins.text(x, 0, "n/a*", ha="center", va="bottom", fontsize=7.5,
                        color=TEXT_MUTED, style="italic")
            continue
        bottom = 0.0
        for d in diameters:
            v = counts.get(d, 0)
            if v > 0:
                ax_ins.bar(x, v, bar_width, bottom=bottom, color=diam_colors[d],
                           edgecolor="white", linewidth=0.6, zorder=3)
            bottom += v

    ax_ins.set_ylabel("Insulated pipe segments [count]")
    ax_ins.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_ins.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_ins.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_ins.spines[spine].set_visible(False)
    ax_ins.spines["left"].set_color(AXIS_COLOR)
    ax_ins.spines["bottom"].set_color(AXIS_COLOR)
    ax_ins.set_title("Insulated pipe segments, by pipe diameter", fontsize=10.5, fontweight="bold",
                      color=TEXT_MUTED, loc="left", pad=4)

    # --- Shared x-axis: scenario labels, group separator, group titles ----
    xs = [positions[sk] for sk in all_scenarios]
    ax_ins.set_xticks(xs)
    ax_ins.set_xticklabels([_scenario_label(sk) for sk in all_scenarios], fontsize=8.5)
    ax_ins.set_xlim(min(xs) - 1.0, max(xs) + 1.0)

    sep_x = (positions[supercritical_order[-1]] + positions[liquid_order[0]]) / 2
    for ax in (ax_boost, ax_ins):
        ax.axvline(sep_x, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)

    # Figure-fraction y for the group labels / footnote, keyed to data-space
    # x on ax_ins -- stable under the fixed subplots_adjust margins above.
    label_trans = transforms.blended_transform_factory(ax_ins.transData, fig.transFigure)
    sc_mid = sum(positions[sk] for sk in supercritical_order) / len(supercritical_order)
    liq_mid = sum(positions[sk] for sk in liquid_order) / len(liquid_order)
    for mid, label in ((sc_mid, "Supercritical phase"), (liq_mid, "Liquid phase")):
        fig.text(mid, 0.085, label, transform=label_trans, ha="center", va="top",
                  fontsize=9.5, fontweight="bold", color=TEXT_MUTED)

    # fig.text() never auto-wraps -- each line is kept short enough for the
    # 11 in canvas at this font size; an overflowing line would otherwise
    # make bbox_inches="tight" balloon the saved image around the overflow.
    footnote_lines = [
        "* pre-insulation-feature benchmark run (bm_sco2 / bm_dense_2); panel A still shows their "
        "boosters (diameter is tracked in both schema versions), panel B has no data to show.",
        "BM = benchmark run.  No ins. = no pipe insulation.  +X% = insulation CAPEX surcharge of X%.  "
        "Diameter shading is ordinal (light = 6\", dark = 42\"), not categorical identity.",
        "Source: analysis/results_data (EUS), 'EUS - Pipes' sheets.",
    ]
    fig.text(0.01, 0.015, "\n".join(footnote_lines), fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom")

    png_path = out_dir / "booster_insulation_by_diameter.png"
    pdf_path = out_dir / "booster_insulation_by_diameter.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector, for \includegraphics{} in LaTeX
    plt.close(fig)

    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")
    print(f"Diameters: {diameters}")
    print(f"Excluded from panel B (no Insulated column): {excluded_insulation}")


if __name__ == "__main__":
    plot_booster_insulation_by_diameter()
