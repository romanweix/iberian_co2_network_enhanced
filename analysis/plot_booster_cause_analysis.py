# plot_booster_cause_analysis.py
#
# Visualizes booster_cause_analysis.py's classification of why boosters
# were needed, straight from the checkpoint-level, two-pass analysis (0
# residual "Inconclusive" pipes here, unlike the older, single-pass,
# results_data-based classification in eus_full_analysis.py /
# plot_booster_vs_insulation.py, which leaves a few unresolved).
#
#   (A) Installed boosters per scenario, stacked by Reason (Pressure only /
#       Temperature only / Both) -- one bar per insulation-surcharge sweep
#       scenario, grouped by phase (supercritical / liquid). Same
#       categorical palette as plot_booster_vs_insulation.py, for visual
#       consistency across the thesis's figures.
#   (B) Reason share [%] by phase x insulation status -- shows how adding
#       insulation shifts a scenario's boosters away from
#       temperature-driven and towards pressure-driven (insulation removes
#       the pipe's own heat loss as a constraint, leaving pressure as the
#       remaining one).
#
# Usage (from the repo root):
#   python -m analysis.plot_booster_cause_analysis

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.booster_cause_analysis import (
    booster_cause_analysis, _checkpoint_files, _scenario_code,
    REASON_ORDER, MODEL_RESULTS_DIR_DEFAULT, UTIL_W_DEFAULT,
)

OUT_DIR = Path("analysis")

# Categorical palette slots 1/2/3 (blue/orange/aqua), same assignment as
# plot_booster_vs_insulation.py's REASON_COLORS.
COLOR_PRESSURE = "#2a78d6"
COLOR_TEMPERATURE = "#eb6834"
COLOR_BOTH = "#1baf7a"
COLOR_INCONCLUSIVE = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#52514e"

REASON_COLORS = {
    "Pressure only": COLOR_PRESSURE,
    "Temperature only": COLOR_TEMPERATURE,
    "Both": COLOR_BOTH,
    "Inconclusive": COLOR_INCONCLUSIVE,
}

# Sweep scenario codes (see analysis/model_results/*_model_checkpoint.dill),
# benchmark-excluded (LPBM/SCBM carry no per-pipe pressure/temperature
# data -- booster_cause_analysis.py drops them upstream), each group
# starting at its "no insulation" baseline.
SUPERCRITICAL_ORDER = ["SCU", "SCI20", "SCI40", "SCI60", "SCI80", "SCI100", "SCI150"]
LIQUID_ORDER = ["LPU", "LPI20", "LPI40", "LPI60"]


def _scenario_label(code: str) -> str:
    if code.endswith("U"):
        return "No ins."
    m = re.search(r"I(\d+)$", code)
    return f"+{m.group(1)}%" if m else code


def _layout_positions(supercritical_order, liquid_order):
    positions = {}
    x = 0.0
    for order in (supercritical_order, liquid_order):
        for i, sk in enumerate(order):
            positions[sk] = x
            x += 1.7 if i == 0 else 1.0
        x += 0.7
    return positions


def plot_booster_cause_analysis(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _checkpoint_files(MODEL_RESULTS_DIR_DEFAULT)
    result = booster_cause_analysis(files, UTIL_W_DEFAULT)
    detail = result["detail"]
    by_phase_ins = result["by_phase_ins"].set_index(["Phase", "Insulated"])

    boostcount = (
        detail.groupby(["Scenario", "Reason"])["Number of boosters"].sum().unstack(fill_value=0.0)
    )
    for r in REASON_ORDER:
        if r not in boostcount.columns:
            boostcount[r] = 0.0

    positions = _layout_positions(SUPERCRITICAL_ORDER, LIQUID_ORDER)
    all_scenarios = SUPERCRITICAL_ORDER + LIQUID_ORDER

    fig, (ax_count, ax_share) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.28},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16)
    bar_width = 0.75

    # --- Panel A: installed boosters per scenario, stacked by reason -----
    for sk in all_scenarios:
        x = positions[sk]
        bottom = 0.0
        for reason in REASON_ORDER:
            v = boostcount.loc[sk, reason] if sk in boostcount.index else 0.0
            if v > 0:
                ax_count.bar(x, v, bar_width, bottom=bottom, color=REASON_COLORS[reason],
                             edgecolor="white", linewidth=0.6, zorder=3)
            bottom += v

    ax_count.set_ylabel("Installed boosters [count]")
    ax_count.set_title("Boosters by scenario", fontsize=10.5, color=TEXT_MUTED, pad=8)
    ax_count.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_count.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_count.spines[spine].set_visible(False)
    ax_count.spines["left"].set_color(AXIS_COLOR)
    ax_count.spines["bottom"].set_color(AXIS_COLOR)

    xs = [positions[sk] for sk in all_scenarios]
    ax_count.set_xticks(xs)
    ax_count.set_xticklabels([_scenario_label(sk) for sk in all_scenarios], fontsize=8.5)
    ax_count.set_xlim(min(xs) - 1.0, max(xs) + 1.0)

    sep_x = (positions[SUPERCRITICAL_ORDER[-1]] + positions[LIQUID_ORDER[0]]) / 2
    ax_count.axvline(sep_x, color=AXIS_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    sc_mid = sum(positions[sk] for sk in SUPERCRITICAL_ORDER) / len(SUPERCRITICAL_ORDER)
    liq_mid = sum(positions[sk] for sk in LIQUID_ORDER) / len(LIQUID_ORDER)
    y_top = ax_count.get_ylim()[1]
    for mid, label in ((sc_mid, "Supercritical"), (liq_mid, "Liquid/dense")):
        ax_count.text(mid, y_top * 1.03, label, ha="center", va="bottom",
                       fontsize=8.5, fontweight="bold", color=TEXT_MUTED)
    ax_count.set_ylim(top=y_top * 1.14)

    # --- Panel B: reason share [%] by phase x insulation status ----------
    groups = [
        ("supercritical", False, "Supercritical\n(no ins.)"),
        ("supercritical", True, "Supercritical\n(insulated)"),
        ("liquid/dense", False, "Liquid/dense\n(no ins.)"),
    ]
    group_x = list(range(len(groups)))
    for gx, (phase, insulated, _) in zip(group_x, groups):
        if (phase, insulated) not in by_phase_ins.index:
            continue
        row = by_phase_ins.loc[(phase, insulated)]
        bottom = 0.0
        for reason in REASON_ORDER:
            v = row.get(f"{reason} [%]", 0.0)
            if v > 0:
                ax_share.bar(gx, v, 0.6, bottom=bottom, color=REASON_COLORS[reason],
                             edgecolor="white", linewidth=0.6, zorder=3)
            bottom += v

    ax_share.set_ylabel("Share of boosted pipes [%]")
    ax_share.set_title("Reason share by phase / insulation", fontsize=10.5, color=TEXT_MUTED, pad=8)
    ax_share.set_ylim(0, 100)
    ax_share.set_xticks(group_x)
    ax_share.set_xticklabels([g[2] for g in groups], fontsize=8.3)
    ax_share.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_share.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_share.spines[spine].set_visible(False)
    ax_share.spines["left"].set_color(AXIS_COLOR)
    ax_share.spines["bottom"].set_color(AXIS_COLOR)

    fig.suptitle("Why boosters were needed: pressure vs. temperature (EUS)",
                 fontsize=13, fontweight="bold", y=0.975)
    legend_handles = [Patch(facecolor=REASON_COLORS[r], label=r) for r in REASON_ORDER
                       if boostcount[r].sum() > 0 or (by_phase_ins[f"{r} [%]"] > 0).any()]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=len(legend_handles), frameon=False, fontsize=8.5)

    fig.text(
        0.01, 0.015,
        "Data: analysis/model_results (EUS, base_utilization), two-pass classification -- "
        "see analysis/booster_cause_analysis.py / .xlsx. Benchmark runs (SCBM/LPBM) excluded: "
        "no per-pipe pressure/temperature/insulation data.",
        fontsize=6.8, color=TEXT_MUTED, ha="left", va="bottom",
    )

    png_path = out_dir / "booster_cause_analysis.png"
    pdf_path = out_dir / "booster_cause_analysis.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector, for \includegraphics{} in LaTeX
    plt.close(fig)

    print(f"Saved {png_path.resolve()}")
    print(f"Saved {pdf_path.resolve()}")


if __name__ == "__main__":
    plot_booster_cause_analysis()
