# replot.py
#
# Regenerate plots from a previously solved & checkpointed model, without
# re-running the (often long) solve in developed_main.py. The checkpoint is
# written by developed_main.py right after a successful solve.
#
# Usage (from the repo root, so the `iberian_co2_network` package resolves):
#   python -m iberian_co2_network.replot

import os
import dill

from iberian_co2_network.developed_plots import plot_network, save_all_plots

CHECKPOINT_PATH = "output_developed_model/model_checkpoint.dill"

with open(CHECKPOINT_PATH, "rb") as f:
    checkpoint = dill.load(f)
m = checkpoint["m"]
DATA = checkpoint["DATA"]

print(f"✔  Loaded checkpoint from {os.path.abspath(CHECKPOINT_PATH)}")

# 1) Regenerate the 3x5 = 15 plots (without showing them on screen)
save_all_plots(m, DATA)

# 2) (Optional) Show a single "reference" plot on screen, matching developed_main.py
plot_network(
    m, DATA,
    scenario="base_utilization",
    year=2050,
    save=True,
    show=True
)

print("\n🎉 Replot completed. Results in /output_developed_model")
