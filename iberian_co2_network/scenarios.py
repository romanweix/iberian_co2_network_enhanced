"""
scenarios.py
~~~~~~~~~~~~
Generates **three deterministic capacity scenarios** for the stochastic CCS
optimisation model.  Capacities are hard‑coded when no baseline file/DataFrame
is provided, following the user's specification:

* Node 4 → 150 Mt / 5 yr (base case)
* Nodes 5&6 → 75 Mt / 5 yr (base case)

Multipliers applied per scenario
===============================
================  ============  ==========
Scenario name     Description   Multiplier
================  ============  ==========
low_utilization   −50 %         0.5
base_utilization  Expected      1.0
high_utilization  +50 %         1.5
================  ============  ==========
Each scenario has probability **1∕3**.

Uncertain nodes default to ``[4, 5, 6]`` and time‑steps to
``[2030, 2035, 2040, 2045, 2050]``.  Capacities at the first step (2030) are
forced to **zero in all scenarios**, matching the problem statement.

If you prefer to override the built‑in baseline, pass either:

* ``cfg["expected_cap_df"]`` – ready‑to‑use DataFrame with columns
  ``node | t | capacity``; or
* ``cfg["expected_capacity_file"]`` – Excel file inside ``cfg["data_dir"]``.

Both options must cover *all* uncertain nodes and time‑steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_scenarios(cfg: dict) -> Tuple[List[str], Dict[str, float], pd.DataFrame]:
    """Create and return the three deterministic capacity scenarios.

    Parameters
    ----------
    cfg
        Configuration dictionary (see ``config.py``). Recognised keys::

            data_dir: str | Path     # Base folder for external data (optional)
            expected_cap_df: DataFrame  # Baseline capacities (optional)
            expected_capacity_file: str # Excel file name (optional)
            nodes_uncertain: list[int]  # Defaults to [4, 5, 6]
            time_steps: list[int]       # Defaults to [2030, 2035, 2040, 2045, 2050]

    Returns
    -------
    scenarios, probs, g_cap
        * ``scenarios`` – list of scenario names
        * ``probs``     – dict *scenario → probability* (all 1∕3)
        * ``g_cap``     – DataFrame indexed by (scenario, node, t) with column
          ``capacity`` (the uncertain parameter :math:`g_{k,t,\omega}`).
    """

    # ---------------------------------------------------------------------
    # Defaults & config
    # ---------------------------------------------------------------------
    nodes: List[int] = cfg.get("nodes_uncertain", [4, 5, 6])
    time_steps: List[int] = cfg.get(
        "time_steps", [2030, 2035, 2040, 2045, 2050]
    )
    first_ts = time_steps[0]

    multipliers = {
        "low_utilization":  0.5,
        "base_utilization": 1.0,
        "high_utilization": 1.5,
    }
    prob = 1.0 / len(multipliers)

    # ---------------------------------------------------------------------
    # Baseline (expected) capacities
    # ---------------------------------------------------------------------
    if "expected_cap_df" in cfg:
        baseline_df = cfg["expected_cap_df"].copy()

    elif "expected_capacity_file" in cfg:
        data_dir = Path(cfg.get("data_dir", "."))
        path = data_dir / cfg["expected_capacity_file"]
        if not path.exists():
            raise FileNotFoundError(path)
        baseline_df = pd.read_excel(path)

    else:
        # ---- Hard‑coded baseline (default) ----
        records: List[dict] = []
        for node in nodes:
            base_cap = 150 if node == 4 else 75
            for t in time_steps:
                cap = 0.0 if t == first_ts else base_cap
                records.append({"node": node, "t": t, "capacity": cap})
        baseline_df = pd.DataFrame(records)

    # Keep only the relevant nodes & timesteps (in case a wider table was given)
    baseline_df = baseline_df[baseline_df["node"].isin(nodes)]
    baseline_df = baseline_df[baseline_df["t"].isin(time_steps)]

    if baseline_df.empty:
        raise ValueError(
            "Baseline capacity data is empty after filtering nodes/time‑steps."
        )

    # Ensure zero capacity at first step
    baseline_df.loc[baseline_df["t"] == first_ts, "capacity"] = 0.0

    # ---------------------------------------------------------------------
    # Build scenarios by scaling the baseline
    # ---------------------------------------------------------------------
    records: List[dict] = []
    for scen, mult in multipliers.items():
        for _, row in baseline_df.iterrows():
            cap = row["capacity"] * mult if row["t"] != first_ts else 0.0
            records.append(
                {
                    "scenario": scen,
                    "node": int(row["node"]),
                    "t": int(row["t"]),
                    "capacity": float(cap),
                    "prob": prob,
                }
            )

    df = pd.DataFrame(records)

    scenarios = list(multipliers.keys())
    probs = {s: prob for s in scenarios}
    g_cap = df.set_index(["scenario", "node", "t"])[["capacity"]]

    return scenarios, probs, g_cap


# ---------------------------------------------------------------------------
# Dunder exports
# ---------------------------------------------------------------------------

__all__ = ["get_scenarios"]
