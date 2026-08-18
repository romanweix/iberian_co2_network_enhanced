# reexport.py
#
# Regenerate the stochastic_results_theta_*.xlsx export from a previously
# solved & checkpointed model, without re-running the (often long) solve in
# developed_main.py. The checkpoint is written by developed_main.py right
# after a successful solve (see also replot.py for the plot equivalent).
#
# Usage (from the repo root, so the `iberian_co2_network` package resolves):
#   python -m iberian_co2_network.reexport

import os
import dill

from iberian_co2_network import developed_solution

CHECKPOINT_PATH = "output_developed_model/model_checkpoint.dill"

with open(CHECKPOINT_PATH, "rb") as f:
    checkpoint = dill.load(f)
m = checkpoint["m"]

print(f"✔  Loaded checkpoint from {os.path.abspath(CHECKPOINT_PATH)}")

developed_solution.export_results(m)

print("\n🎉 Re-export completed. Results in /output_developed_model")
