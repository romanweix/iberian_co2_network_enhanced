# booster_cause_analysis.py
#
# For every installed pipe that gets a booster station, classifies WHY the
# booster was needed -- Pressure only / Temperature only / Both -- and
# writes the per-pipe detail plus cross-scenario summary pivots to
# analysis/booster_cause_analysis.xlsx.
#
# Reads directly from the solved model checkpoints in analysis/model_results/
# (one dill-pickled {"m": <pyomo ConcreteModel>, "DATA": <dict>} per
# scenario, written by developed_main.py after a successful solve -- see
# iberian_co2_network/reexport.py for the same load pattern), not from the
# exported results_data/*.xlsx workbooks. That matters because it lets a
# second pass query the model's own constraints for the handful of pipes
# whose own pressure/temperature snapshot looks comfortable and yet still
# got boosted -- a node-sharing effect invisible to any single-pipe view
# (see resolve_inconclusive()).
#
# Usage (from the repo root):
#   python -m analysis.booster_cause_analysis
#   python -m analysis.booster_cause_analysis --model-results-dir analysis/model_results --util base_utilization

import argparse
from pathlib import Path

import dill
import pandas as pd
from pyomo.environ import value

from iberian_co2_network import developed_solution as ds
from analysis.eus_full_analysis import _autosize, _style_header

OUT_PATH_DEFAULT = Path("analysis/booster_cause_analysis.xlsx")
MODEL_RESULTS_DIR_DEFAULT = Path("analysis/model_results")

# LUS/EUS/HUS utilization-scenario labels are m.W set members; EUS (base
# case) is what EUS_full_analysis.xlsx's 'Booster reason' sheet also uses.
UTIL_W_DEFAULT = "base_utilization"

# Legacy benchmark checkpoints that predate the pipeline-insulation /
# temperature feature: create_pipe_summary() returns no per-pipe
# temperature or insulation columns for them at all, so there is nothing to
# classify a "cause" from.
EXCLUDED_SCENARIOS = {"LPBM", "SCBM"}

# Node tolerance bands a non-dominant supplying pipe must stay within (see
# NodePressTol_LB/UB and NodeTempTol_LB/UB in developed_model.py).
NODE_PRESS_TOL_BAR = 30.0
NODE_TEMP_TOL_C = 5.0

# Absolute tolerance for treating a margin as exactly zero rather than a
# floating-point artifact of the LP solution.
TOL = 1e-6

# How close (in bar / degC) a pipe's own pass-1 margin must be to zero to be
# read as "near-threshold" in resolve_inconclusive()'s fallback.
NEAR_THRESHOLD_TOL = 1.0

REASON_ORDER = ["Pressure only", "Temperature only", "Both", "Inconclusive"]

DETAIL_COLS = [
    "Scenario", "Phase", "Pipe ID", "Diameter [inch]", "Insulated",
    "Number of boosters", "Installation Year", "Booster installation year",
    "Pressure at highest point [bar]", "Lowest pressure [bar]", "p_min [bar]",
    "Temperature at critical pressure point [°C]", "Lowest temperature [°C]", "theta_min [°C]",
    "Reason", "Resolution note",
]


def _scenario_code(path: Path) -> str:
    return path.stem[: -len("_model_checkpoint")] if path.stem.endswith("_model_checkpoint") else path.stem


def _phase_label(code: str) -> str:
    if code.startswith("LP"):
        return "liquid/dense"
    if code.startswith("SC"):
        return "supercritical"
    return "unknown"


def _checkpoint_files(model_results_dir: Path) -> list[Path]:
    files = sorted(model_results_dir.glob("*_model_checkpoint.dill"))
    if not files:
        raise FileNotFoundError(f"No *_model_checkpoint.dill files found in {model_results_dir}")
    return [f for f in files if _scenario_code(f) not in EXCLUDED_SCENARIOS]


def classify_reason(row: pd.Series, p_min: float, theta_min: float) -> tuple[str, bool, bool]:
    """Pass 1: read the reason straight off the pipe's own installation-year
    snapshot, as exported by create_pipe_summary() -- a reported pressure or
    temperature already below p_min/theta_min directly signals that
    dimension needed the booster."""
    p_flag = (row["Pressure at highest point [bar]"] < p_min - TOL) or \
             (row["Lowest pressure [bar]"] < p_min - TOL)
    if pd.isna(row["Temperature at critical pressure point [°C]"]):
        t_flag = False
    else:
        t_flag = (row["Temperature at critical pressure point [°C]"] < theta_min - TOL) or \
                 (row["Lowest temperature [°C]"] < theta_min - TOL)

    if p_flag and t_flag:
        return "Both", p_flag, t_flag
    if p_flag:
        return "Pressure only", p_flag, t_flag
    if t_flag:
        return "Temperature only", p_flag, t_flag
    return "Inconclusive", p_flag, t_flag


def _pipe_diam_u(m, p, w):
    if p in m.P_on:
        for d in m.D:
            for u in m.U:
                if value(ds._b_diam_on(m, p, d, u, w)) > 0.5:
                    return d, u
        return None, None
    for d in m.D:
        if value(ds._b_diam_off(m, p, d, w)) > 0.5:
            return d, None
    return None, None


def _nboost(m, p, w, t):
    if p in m.P_on:
        return value(ds._brep_on1(m, p, t, w)) + value(ds._brep_on2(m, p, t, w))
    return value(ds._brep_off(m, p, t, w))


def _own_requirement(m, p, w, t, p_min, theta_min):
    """This pipe's own minimum origin pressure/temperature at year t (its
    HighPointMinPressure / PmaxPointMinTemp bound), net of its own
    booster's contribution -- i.e. what its upstream node must supply for
    this pipe's own constraints to hold."""
    d, u = _pipe_diam_u(m, p, w)
    if d is None:
        return None, None
    nboost = _nboost(m, p, w, t)
    Dpi = value(m.delta_p_boost)
    if p in m.P_on:
        Dth = value(m.delta_theta_boost)
        req_p = p_min + value(m.Pi_pmax[p, d, u]) - Dpi * nboost
        req_t = theta_min + value(m.Theta_pmax[p, d, u]) - Dth * nboost
    else:
        req_p = p_min + value(m.dP_frict_far[p, d]) - Dpi * nboost
        req_t = None
    return req_p, req_t


def _unboosted_dest(m, p, w, t):
    """This pipe's own pi_dest/theta_dest at year t with its own booster's
    contribution subtracted out -- what the destination would be if this
    pipe's booster did not exist."""
    nboost = _nboost(m, p, w, t)
    pi_dest = value(m.pi_dest[p, t, w]) - value(m.delta_p_boost) * nboost
    theta_dest = value(m.theta_dest[p, t, w]) - value(m.delta_theta_boost) * nboost if p in m.P_on else None
    return pi_dest, theta_dest


def _near_threshold_fallback(row, p_min, theta_min):
    p_margin = min(row["Pressure at highest point [bar]"], row["Lowest pressure [bar]"]) - p_min
    if pd.isna(row["Temperature at critical pressure point [°C]"]):
        t_margin = None
    else:
        t_margin = min(row["Temperature at critical pressure point [°C]"], row["Lowest temperature [°C]"]) - theta_min

    candidates = []
    yr = row["Booster installation year"]
    if p_margin <= NEAR_THRESHOLD_TOL:
        candidates.append(("Pressure only", p_margin,
                            f"Near-threshold: pressure margin is only {p_margin:.2f} bar "
                            f"above p_min at the install/booster year {int(yr)}."))
    if t_margin is not None and t_margin <= NEAR_THRESHOLD_TOL:
        candidates.append(("Temperature only", t_margin,
                            f"Near-threshold: Lowest temperature is only {t_margin:.2f} above "
                            f"its limit at the (single) install/booster year {int(yr)}."))
    if not candidates:
        return "Inconclusive", None
    candidates.sort(key=lambda c: c[1])
    reason, _, note = candidates[0]
    return reason, note


def resolve_inconclusive(m, p, w, install_year, booster_year, p_min, theta_min):
    """Pass 2: for a pipe whose own snapshot showed comfortable margin on
    both dimensions and yet still got boosted, check the model's own
    node-sharing constraints directly -- pi_node/theta_node is a single
    variable shared by every pipe touching a node, so a pipe can be forced
    to boost purely to satisfy a *different* pipe (if dominant at its
    downstream node) or to stay within the +/-30 bar / +/-5°C tolerance
    band around that node's level (if non-dominant)."""
    if booster_year is None:
        return "Inconclusive", None

    t = int(booster_year)
    j = value(m.end[p])
    is_on = p in m.P_on
    is_dominant = (value(m.y_on[p, t, w]) if is_on else value(m.y_off[p, t, w])) > 0.5
    pi_dest_unboosted, theta_dest_unboosted = _unboosted_dest(m, p, w, t)

    p_violation = t_violation = False
    note = None

    if int(booster_year) != int(install_year):
        if is_dominant:
            for p2 in m.P:
                if p2 == p or value(m.start[p2]) != j:
                    continue
                active2 = value(m.act_on[p2, t, w]) if p2 in m.P_on else value(m.act_off[p2, t, w])
                if active2 < 0.5:
                    continue
                req_p, req_t = _own_requirement(m, p2, w, t, p_min, theta_min)
                if req_p is None:
                    continue
                hit_p = pi_dest_unboosted < req_p - TOL
                hit_t = (req_t is not None and theta_dest_unboosted is not None
                         and theta_dest_unboosted < req_t - TOL)
                if hit_p or hit_t:
                    p_violation, t_violation = hit_p, hit_t
                    note = (
                        f"Shared-node effect: this pipe is the dominant supplier at node {j}; "
                        f"without its booster, downstream pipe {p2} would see "
                        f"pi_node={pi_dest_unboosted:.0f} bar"
                        + (f" / theta_node={theta_dest_unboosted:.0f}°C" if theta_dest_unboosted is not None else "")
                        + f" at that node, short of the {req_p:.0f} bar"
                        + (f" / {req_t:.0f}°C" if req_t is not None else "")
                        + " it requires."
                    )
                    break
        else:
            pi_node = value(m.pi_node[j, t, w])
            theta_node = value(m.theta_node[j, t, w]) if is_on else None
            p_violation = pi_dest_unboosted < pi_node - NODE_PRESS_TOL_BAR - TOL
            t_violation = (theta_node is not None and theta_dest_unboosted is not None
                            and theta_dest_unboosted < theta_node - NODE_TEMP_TOL_C - TOL)
            if p_violation or t_violation:
                note = (
                    f"Shared-node effect: this pipe is a non-dominant supplier at node {j}; "
                    f"without its booster its own pi_dest/theta_dest ({pi_dest_unboosted:.0f} bar"
                    + (f" / {theta_dest_unboosted:.0f}°C" if theta_dest_unboosted is not None else "")
                    + f") would fall outside the required +/-{NODE_PRESS_TOL_BAR:.0f} bar"
                    + (f" / +/-{NODE_TEMP_TOL_C:.0f}°C" if theta_node is not None else "")
                    + f" tolerance band around the node's {pi_node:.0f} bar"
                    + (f" / {theta_node:.0f}°C" if theta_node is not None else "") + "."
                )

    if p_violation or t_violation:
        reason = "Both" if (p_violation and t_violation) else ("Pressure only" if p_violation else "Temperature only")
        return reason, note

    return None, None  # caller falls back to the near-threshold reading


def _pipe_cause_rows(m, DATA, scenario_code: str, phase: str, util_w: str) -> list[dict]:
    p_min = DATA["p_min"]
    theta_min = DATA["theta_min"]
    df = ds.create_pipe_summary(m, util_w)
    boosted = df[(df["Installed"] == 1) & (df["Number of boosters"].fillna(0) > 0)]

    rows = []
    for _, r in boosted.iterrows():
        reason, p_flag, t_flag = classify_reason(r, p_min, theta_min)
        note = None
        if reason == "Inconclusive":
            resolved, note = resolve_inconclusive(
                m, r["Pipe ID"], util_w, r["Installation Year"], r["Booster installation year"], p_min, theta_min
            )
            if resolved is None:
                resolved, note = _near_threshold_fallback(r, p_min, theta_min)
            reason = resolved

        rows.append({
            "Scenario": scenario_code,
            "Phase": phase,
            "Pipe ID": r["Pipe ID"],
            "Diameter [inch]": r["Diameter [inch]"],
            "Insulated": bool(r["Insulated"]) if pd.notna(r["Insulated"]) else False,
            "Number of boosters": r["Number of boosters"],
            "Installation Year": r["Installation Year"],
            "Booster installation year": r["Booster installation year"],
            "Pressure at highest point [bar]": r["Pressure at highest point [bar]"],
            "Lowest pressure [bar]": r["Lowest pressure [bar]"],
            "p_min [bar]": p_min,
            "Temperature at critical pressure point [°C]": r["Temperature at critical pressure point [°C]"],
            "Lowest temperature [°C]": r["Lowest temperature [°C]"],
            "theta_min [°C]": theta_min,
            "Reason": reason,
            "Resolution note": note,
        })
    return rows


def _summary_pivot(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = detail.groupby(group_cols)["Reason"].value_counts().unstack(fill_value=0)
    for r in REASON_ORDER:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[REASON_ORDER]
    counts.columns = [f"{c} [n]" for c in counts.columns]
    total = counts.sum(axis=1)
    pct = counts.div(total, axis=0) * 100.0
    pct.columns = [c.replace(" [n]", " [%]") for c in counts.columns]
    out = pd.concat([counts, pct], axis=1)
    out["Total [n]"] = total
    return out.reset_index()


def booster_cause_analysis(files: list[Path], util_w: str) -> dict[str, pd.DataFrame]:
    all_rows = []
    for f in files:
        code = _scenario_code(f)
        phase = _phase_label(code)
        with open(f, "rb") as fh:
            checkpoint = dill.load(fh)
        all_rows.extend(_pipe_cause_rows(checkpoint["m"], checkpoint["DATA"], code, phase, util_w))

    detail = pd.DataFrame(all_rows, columns=DETAIL_COLS)
    detail["Group"] = "All scenarios"

    overall = _summary_pivot(detail, ["Group"])
    by_phase = _summary_pivot(detail, ["Phase"])
    by_phase_ins = _summary_pivot(detail, ["Phase", "Insulated"])
    detail = detail.drop(columns=["Group"])

    return {
        "detail": detail,
        "overall": overall,
        "by_phase": by_phase,
        "by_phase_ins": by_phase_ins,
    }


NOTES = [
    "Reason is derived from each boosted pipe's simulated pressure/temperature at its "
    "installation-year snapshot (create_pipe_summary() in developed_solution.py), read "
    "directly from each scenario's solved model checkpoint (analysis/model_results/).",
    "p_min and theta_min are read per scenario from the checkpoint's own DATA['p_min']/"
    "DATA['theta_min'], not hardcoded.",
    "Reason is resolved in two passes. Pass 1 (classify_reason()) checks the pipe's own "
    "installation-year snapshot, as above. Any pipe left unresolved there -- comfortable "
    "on both dimensions at that snapshot, yet boosted -- is resolved in pass 2 "
    "(resolve_inconclusive()) directly against the solved model's node-sharing "
    "constraints (NodePressTol/NodeTempTol, dominant-pipe selection), falling back to a "
    "near-threshold reading of the pipe's own snapshot if that finds nothing either; see "
    "the 'Resolution note' column in 'Per-pipe detail' for which mechanism applied.",
    "Insulated = whether the pipe itself carries the U=0.43 W/m^2/K insulated wall option "
    "(vs. U=2.0 uninsulated); offshore pipes are never insulated in this model and always "
    "show False.",
    "Phase is read from the scenario code prefix: LP* = liquid/dense phase, SC* = supercritical phase.",
    f"Excluded (legacy pre-insulation-feature checkpoints, no per-pipe pressure/"
    f"temperature/insulation data): {', '.join(sorted(EXCLUDED_SCENARIOS))}.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-results-dir", type=Path, default=MODEL_RESULTS_DIR_DEFAULT)
    parser.add_argument("--util", default=UTIL_W_DEFAULT,
                         help="m.W scenario key (low_utilization/base_utilization/high_utilization)")
    parser.add_argument("--out", type=Path, default=OUT_PATH_DEFAULT)
    args = parser.parse_args()

    files = _checkpoint_files(args.model_results_dir)
    result = booster_cause_analysis(files, args.util)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        result["overall"].to_excel(writer, sheet_name="Reason - overall", index=False)
        result["by_phase"].to_excel(writer, sheet_name="Reason x Phase", index=False)
        result["by_phase_ins"].to_excel(writer, sheet_name="Reason x Phase x Insulated", index=False)
        result["detail"].to_excel(writer, sheet_name="Per-pipe detail", index=False)
        pd.DataFrame({"Note": NOTES}).to_excel(writer, sheet_name="Notes", index=False, header=False)

        for sheet_name, df in [
            ("Reason - overall", result["overall"]),
            ("Reason x Phase", result["by_phase"]),
            ("Reason x Phase x Insulated", result["by_phase_ins"]),
            ("Per-pipe detail", result["detail"]),
        ]:
            ws = writer.sheets[sheet_name]
            _style_header(ws, len(df.columns))
            _autosize(ws)

    print(f"✔  Cause analysis workbook written to {args.out.resolve()}")
    print("\n--- Reason, all scenarios ---")
    print(result["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
