# plot_booster_diameter_barchart.py
#
# Generates the grouped bar chart of booster stations by pipeline diameter
# (EUS reference run vs. sCO2/liquid-phase no-insulation scenarios), as a
# standalone alternative to the pgfplots version in
# analysis/booster_diameter_barchart.tex -- same data, plain PNG/PDF output
# for direct \includegraphics{} use in the thesis.
#
# Usage:
#   python -m analysis.plot_booster_diameter_barchart

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DIAMETERS = ["6", "10", "14", "18", "22", "26", "30", "34", "38", "42"]

# Same data as analysis/booster_diameter_comparison_table.tex
BOOSTERS = {
    "BM":            [0, 0, 0, 0, 1, 1, 1, 1, 2, 0],
    "sCO$_2$ No Ins.": [0, 0, 0, 1, 0, 1, 3, 1, 1, 1],
    "Dense No Ins.":     [0, 0, 0, 1, 1, 2, 1, 1, 1, 0],
}


def plot_booster_diameter_barchart(out_dir: Path = Path("analysis")):
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(DIAMETERS))
    n_series = len(BOOSTERS)
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, values) in enumerate(BOOSTERS.items()):
        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width, label=label)
        ax.bar_label(bars, fontsize=7, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels(DIAMETERS)
    ax.set_xlabel("Pipeline diameter [inch]")
    ax.set_ylabel("Number of booster stations")
    ax.set_title("Booster stations by pipeline diameter")
    ax.set_ylim(0, max(v for vals in BOOSTERS.values() for v in vals) + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=n_series, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    png_path = out_dir / "booster_diameter_barchart.png"
    pdf_path = out_dir / "booster_diameter_barchart.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector, for \includegraphics{} in LaTeX
    plt.close(fig)

    print(f"✔  Saved {png_path.resolve()}")
    print(f"✔  Saved {pdf_path.resolve()}")


if __name__ == "__main__":
    plot_booster_diameter_barchart()
