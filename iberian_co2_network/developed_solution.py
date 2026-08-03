# developed_solution.py

"""
Utility functions to extract results from the Pyomo model and export them to Excel
spreadsheets (scenario-aware).

All monetary figures are assumed to already be in present-value terms (M€).
"""

from pathlib import Path
from typing import Sequence

import pandas as pd
from pyomo.environ import value
from openpyxl.styles import Font, Alignment, Border, Side

# ---------------------------------------------------------------------------
# Output folder and global settings
# ---------------------------------------------------------------------------

_OUTDIR = Path("output_developed_model")
_OUTDIR.mkdir(exist_ok=True)

years_per_step = 5  # Number of years per time step in the model


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _first(iterable: Sequence):
    """Return the first item of *iterable* or ``None`` if empty."""
    return next(iter(iterable), None)


def _diameter_to_inch(m, d):
    """
    If a Param diam_inch exists -> use it.
    Otherwise assume *d* already contains the inch value (string or int).
    """
    if d is None:
        return ""
    if hasattr(m, "diam_inch"):
        return value(m.diam_inch[d])
    try:
        return int(d)
    except (ValueError, TypeError):
        return d


def _years_from_T(m) -> list[int]:
    """Return sorted integer years from model set T."""
    return sorted(int(t) for t in m.T)


def _t_from_year(m, yr: int):
    """Return the T element matching integer year *yr*, or None."""
    return _first([t for t in m.T if int(t) == int(yr)])

def _format_header_row(ws):
    # Header style (bold + centered + thin black border)
    header_font = Font(bold=True)
    header_alignment = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="000000")
    header_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:  # first row
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = header_border


# ---------------------------------------------------------------------------
# Effective P1/P2 selectors (updated to match the new model)
#   - P1 variables are scenario-independent (no omega index)
#   - P2 variables are scenario-dependent (omega index)
# ---------------------------------------------------------------------------

def _z_on(m, p, t, w):
    """Effective build binary for onshore pipe p at time t in scenario w."""
    return m.z_on_P1[p, t] if p in m.P1_on else m.z_on_P2[p, t, w]


def _z_off(m, p, t, w):
    """Effective build binary for offshore pipe p at time t in scenario w."""
    return m.z_off_P1[p, t] if p in m.P1_off else m.z_off_P2[p, t, w]


def _b_diam_on(m, p, d, w):
    """Effective diameter selection binary for onshore pipe p and diameter d in scenario w."""
    return m.b_diam_on_P1[p, d] if p in m.P1_on else m.b_diam_on_P2[p, d, w]


def _b_diam_off(m, p, d, w):
    """Effective diameter selection binary for offshore pipe p and diameter d in scenario w."""
    return m.b_diam_off_P1[p, d] if p in m.P1_off else m.b_diam_off_P2[p, d, w]


def _brep_on1(m, p, t, w):
    """Effective booster-1 binary for onshore pipe p at time t in scenario w."""
    return m.brep_on1_P1[p, t] if p in m.P1_on else m.brep_on1_P2[p, t, w]


def _brep_on2(m, p, t, w):
    """Effective booster-2 binary for onshore pipe p at time t in scenario w."""
    return m.brep_on2_P1[p, t] if p in m.P1_on else m.brep_on2_P2[p, t, w]


def _brep_off(m, p, t, w):
    """Effective booster binary for offshore pipe p at time t in scenario w."""
    return m.brep_off_P1[p, t] if p in m.P1_off else m.brep_off_P2[p, t, w]


# ---------------------------------------------------------------------------
# 1) PIPE SUMMARY TABLE (per scenario)
# ---------------------------------------------------------------------------

def create_pipe_summary(m, w, years_flow=None) -> pd.DataFrame:
    """
    Per-pipeline summary for scenario w (omega):
      - Automatically detects P1 vs P2 and uses the correct variables.
      - Flows and pressures are indexed by (p,t,w).
      - CAPEX includes P1 (no omega) plus P2 (omega, for the chosen scenario).
    """
    if years_flow is None:
        years_flow = _years_from_T(m)

    P_on = set(m.P_on)
    P_off = set(m.P_off)
    P1_on, P2_on = set(m.P1_on), set(m.P2_on)
    P1_off, P2_off = set(m.P1_off), set(m.P2_off)

    dp_boost = value(m.delta_p_boost) if hasattr(m, "delta_p_boost") else 50.0

    records = []

    for p in list(P_on) + list(P_off):
        is_on = p in P_on

        # Lengths
        L = value(m.L_on[p]) if is_on else value(m.L_off[p])
        Lh = value(m.Lh_on[p]) if is_on else value(m.Lh_off[p])

        # Installation year(s): any build-binary > 0.5
        install_years = []
        for t in m.T:
            zval = value(_z_on(m, p, t, w)) if is_on else value(_z_off(m, p, t, w))
            if zval > 0.5:
                install_years.append(int(t))
        install_years.sort()
        installed = 1 if install_years else 0
        year = install_years[0] if installed else ""

        # Selected diameter
        diam_selected = None
        for d in m.D:
            bd = value(_b_diam_on(m, p, d, w)) if is_on else value(_b_diam_off(m, p, d, w))
            if bd > 0.5:
                diam_selected = d
                break
        diam_inch = _diameter_to_inch(m, diam_selected) if diam_selected is not None else ""

        # Boosters (max over horizon)
        n_boost = 0
        if is_on:
            for t in m.T:
                n_boost = max(
                    n_boost,
                    int(round(value(_brep_on1(m, p, t, w)))) + int(round(value(_brep_on2(m, p, t, w))))
                )
        else:
            for t in m.T:
                n_boost = max(n_boost, int(round(value(_brep_off(m, p, t, w)))))

        # Pressures at installation year (use the actual model values)
        pi_init = pi_high = pi_final = pi_lowest = None
        if installed and year != "":
            t_match = _t_from_year(m, year)
            if t_match is not None:
                pi_init = value(m.pi_orig[p, t_match, w])
                pi_final = value(m.pi_dest[p, t_match, w])

                # Compute an approximate "high point" pressure (only meaningful for onshore)
                if is_on and diam_selected is not None:
                    dp_fric_high = value(m.dP_frict_high[p, diam_selected])
                    dp_elev_high = value(m.dP_elev_high[p])
                    pi_high = pi_init - (dp_fric_high + dp_elev_high)

                # Compute an approximate "lowest pressure" using far-drop and (optional) high point
                if diam_selected is not None:
                    if is_on:
                        dp_fric_far = value(m.dP_frict_far[p, diam_selected])
                        dp_elev_far = value(m.dP_elev_far[p])
                        pi_far_no_boost = pi_init - (dp_fric_far + dp_elev_far)
                    else:
                        dp_fric_far = value(m.dP_frict_far[p, diam_selected])
                        # Offshore far-elevation is already embedded (as in the model), so keep 0 here
                        pi_far_no_boost = pi_init - dp_fric_far

                    # If you want an "unboosted" far pressure, add a column below; here we keep lowest approx.
                    candidates = [pi_far_no_boost]
                    if pi_high is not None:
                        candidates.append(pi_high)
                    pi_lowest = min(candidates) if candidates else None

        # Present-value CAPEX: P1 counted once (no omega); P2 for scenario w
        capex = 0.0
        if is_on:
            if p in P1_on:
                capex += sum(value(m.c_pipe_on_P1[p, t]) for t in m.T)
            if p in P2_on:
                capex += sum(value(m.c_pipe_on_P2[p, t, w]) for t in m.T)
        else:
            if p in P1_off:
                capex += sum(value(m.c_pipe_off_P1[p, t]) for t in m.T)
            if p in P2_off:
                capex += sum(value(m.c_pipe_off_P2[p, t, w]) for t in m.T)

        # Connections and node types
        i_node = value(m.start[p])
        j_node = value(m.end[p])
        node_conn = (i_node, j_node)
        node_types = (value(m.n1_type[p]), value(m.n2_type[p]))

        # Flows in selected years (scenario w)
        flow_cols = {}
        for yr in years_flow:
            t_match = _t_from_year(m, yr)
            if t_match is None:
                flow_cols[f"Flow in {yr}"] = None
            else:
                q = value(m.q_on[p, t_match, w]) if is_on else value(m.q_off[p, t_match, w])
                flow_cols[f"Flow in {yr}"] = q if installed else None

        rec = {
            "Pipe ID": p,
            "Installed": installed,
            "Diameter [inch]": diam_inch,
            "Distance until highest point [km]": Lh,
            "Longitude [km]": L,
            "Node connections": node_conn,
            "Node types": node_types,
            "Initial pressure [bar]": pi_init if installed else None,
            "Pressure at highest point [bar]": pi_high if installed else None,
            "Final pressure [bar]": pi_final if installed else None,
            "Lowest pressure [bar]": pi_lowest if installed else None,
            "Number of boosters": n_boost if installed else None,
            "Installation Year": year if installed else None,
            "Present Value Cost [M€]": capex if installed else None,
            "Booster delta pressure [bar]": dp_boost if installed else None,
        }
        rec.update(flow_cols)
        records.append(rec)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 2) NODE SUMMARY TABLE (per scenario)
# ---------------------------------------------------------------------------

def _node_type(m, i):
    """Return a single-letter node type based on membership in model sets."""
    if i in m.E:
        return "E"
    if i in m.S:
        return "S"
    if i in m.K:
        return "K"
    if i in m.A:
        return "A"
    if i in m.M:
        return "M"
    return "?"


def create_node_summary(m, w: str, year: int | None = None) -> pd.DataFrame:
    """One row per node for scenario *w* and year *year* (last year if None)."""
    if year is None:
        year = max(int(t) for t in m.T)

    t_match = _t_from_year(m, year)
    if t_match is None:
        raise ValueError(f"Year {year} not found in model set T")

    records = []
    for i in m.N:
        ntype = _node_type(m, i)
        rec = {
            "Node ID": i,
            "Node type": ntype,
            "Emitted CO2 [Mt]": value(m.g[i, t_match]) if (ntype == "E" and (i, t_match) in m.g) else (value(m.g[i, t_match]) if ntype == "E" else None),
            "Captured CO2 [Mt]": value(m.qcap[i, t_match, w]) if ntype == "E" else None,
            "Stored CO2 [Mt]": value(m.qstore[i, t_match, w]) if ntype == "S" else None,
            "Used CO2 [Mt]": value(m.qsale[i, t_match, w]) if ntype == "K" else None,
            "Pressure [bar]": value(m.pi_node[i, t_match, w]),
        }
        records.append(rec)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 3) SINK CAPACITY EVOLUTION (per scenario)
# ---------------------------------------------------------------------------

def create_sink_evolution(m, w: str, years_flow=None) -> pd.DataFrame:
    """One row per sink node; reports remaining capacity by year for scenario w."""
    if years_flow is None:
        years_flow = _years_from_T(m)

    records = []
    for s in m.S:
        rec = {
            "Sink node": s,
            "Initial capacity [Mt]": value(m.cap_store[s]),
        }

        for yr in years_flow:
            t_match = _t_from_year(m, yr)
            rec[f"{yr} capacity [Mt]"] = value(m.alpha[s, t_match, w]) if t_match is not None else None

        # Final capacity after last injection (alpha - qstore) at the last year in T
        last_year = max(years_flow)
        t_last = _t_from_year(m, last_year)
        if t_last is not None:
            rec["Final capacity [Mt]"] = value(m.alpha[s, t_last, w]) - value(m.qstore[s, t_last, w])
        else:
            rec["Final capacity [Mt]"] = None

        records.append(rec)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 4) UTILIZATION EVOLUTION (per scenario)
# ---------------------------------------------------------------------------

def create_util_evolution(m, w: str, years_flow=None) -> pd.DataFrame:
    """One row per utilization node K; reports scenario capacity (g_cap) and sales by year."""
    if years_flow is None:
        years_flow = _years_from_T(m)

    records = []
    for k in m.K:
        cap_max = max(value(m.g_cap[k, t, w]) for t in m.T)
        rec = {
            "Utilization node": k,
            "Utilization capacity [Mt]": cap_max,
        }
        for yr in years_flow:
            t_match = _t_from_year(m, yr)
            rec[f"{yr} utilization [Mt]"] = value(m.qsale[k, t_match, w]) if t_match is not None else None

        records.append(rec)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 5) COST BREAKDOWN TABLE (per scenario)
# ---------------------------------------------------------------------------

def create_cost_breakdown(m, w: str, years=None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      Concept · 2030 · 2035 · … · Total

    Costs are computed per scenario (not expected value).
    """
    if years is None:
        years = _years_from_T(m)

    def _t(yr):
        return _t_from_year(m, yr)

    # --- yearly components (scenario w) -----------------------------------

    def capex_pipe_on_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return (
            sum(value(m.c_pipe_on_P1[p, t]) for p in m.P1_on)
            + sum(value(m.c_pipe_on_P2[p, t, w]) for p in m.P2_on)
        )

    def capex_pipe_off_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return (
            sum(value(m.c_pipe_off_P1[p, t]) for p in m.P1_off)
            + sum(value(m.c_pipe_off_P2[p, t, w]) for p in m.P2_off)
        )

    def capex_init_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return sum(value(m.cins_init[t]) * value(m.qcap_design[e]) * value(m.z_init[e, t, w]) for e in m.E)

    def capex_boost_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return (
            # P1 booster CAPEX (no scenario index)
            sum(value(m.c_boost_on1_P1[p, t]) + value(m.c_boost_on2_P1[p, t]) for p in m.P1_on)
            + sum(value(m.c_boost_off_P1[p, t]) for p in m.P1_off)
            # P2 booster CAPEX (scenario-indexed)
            + sum(value(m.c_boost_on1_P2[p, t, w]) + value(m.c_boost_on2_P2[p, t, w]) for p in m.P2_on)
            + sum(value(m.c_boost_off_P2[p, t, w]) for p in m.P2_off)
        )

    def capture_cost_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return sum(value(m.capture_cost[t]) * value(m.qcap[e, t, w]) for e in m.E)

    def injection_cost_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return sum(value(m.injection_cost[t]) * value(m.qstore[s, t, w]) for s in m.S)

    def opex_pipe_on_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        total = 0.0
        for p in m.P_on:
            act = value(m.act_on[p, t, w])
            if act <= 1e-12:
                continue
            for d in m.D:
                bd = value(_b_diam_on(m, p, d, w))
                if bd <= 1e-12:
                    continue
                total += (
                    years_per_step
                    * value(m.cop_on[d, t])
                    * value(m.L_on[p])
                    * value(m.pen_city[p])
                    * value(m.pen_slope[p])
                    * bd
                    * act
                )
        return total

    def opex_pipe_off_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        total = 0.0
        for p in m.P_off:
            act = value(m.act_off[p, t, w])
            if act <= 1e-12:
                continue
            for d in m.D:
                bd = value(_b_diam_off(m, p, d, w))
                if bd <= 1e-12:
                    continue
                total += (
                    years_per_step
                    * value(m.cop_off[d, t])
                    * value(m.L_off[p])
                    * value(m.pen_city[p])
                    * value(m.pen_slope[p])
                    * bd
                    * act
                )
        return total

    def ship_cost_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return sum(
            value(m.nship[p, t, w]) * (value(m.ship_fixed_cost[t]) + value(m.ship_fuel_cost[t]) * value(m.L_off[p]))
            for p in m.P_off
        )

    def opex_init_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return sum(value(m.cop_init[t]) * value(m.qcap_design[e]) * value(m.act_init[e, t, w]) for e in m.E)

    def el_init_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return sum(value(m.cel_init[t]) * value(m.qcap[e, t, w]) for e in m.E)

    def opex_boost_year(yr):
        """
        Booster OPEX consistent with the UPDATED model objective:
        cop_boost[t] * (installed booster capacity)
        """
        t = _t(yr)
        if t is None:
            return 0.0

        return value(m.cop_boost[t]) * (
            # P1 (no scenario index)
            sum(value(m.w_boostcap_on1_P1[p, t]) + value(m.w_boostcap_on2_P1[p, t]) for p in m.P1_on)
            + sum(value(m.w_boostcap_off_P1[p, t]) for p in m.P1_off)
            # P2 (scenario-indexed)
            + sum(value(m.w_boostcap_on1_P2[p, t, w]) + value(m.w_boostcap_on2_P2[p, t, w]) for p in m.P2_on)
            + sum(value(m.w_boostcap_off_P2[p, t, w]) for p in m.P2_off)
        )

    def el_boost_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        return value(m.cel_boost[t]) * (
            sum(value(m.w_boost_on1[p, t, w]) for p in m.P_on)
            + sum(value(m.w_boost_on2[p, t, w]) for p in m.P_on)
            + sum(value(m.w_boost_off[p, t, w]) for p in m.P_off)
        )

    def penalty_uncap_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        uncaptured = (
            sum(value(m.g[e, t]) for e in m.E)
            - sum(value(m.qstore[s, t, w]) for s in m.S)
            - sum(value(m.qsale[k, t, w]) for k in m.K)
        )
        return uncaptured * value(m.allow_price[t])

    def sale_revenue_year(yr):
        t = _t(yr)
        if t is None:
            return 0.0
        # Negative cost (revenue)
        return -sum(value(m.qsale[k, t, w]) * value(m.selling_revenue[t]) for k in m.K)

    def total_by_year(func):
        out = {}
        for yr in years:
            out[yr] = func(yr)
        return out

    rows = [
        ("Capture cost", total_by_year(capture_cost_year)),
        ("Injection cost", total_by_year(injection_cost_year)),
        ("CAPEX onshore pipe", total_by_year(capex_pipe_on_year)),
        ("CAPEX offshore pipe", total_by_year(capex_pipe_off_year)),
        ("CAPEX initial boosting stations", total_by_year(capex_init_year)),
        ("CAPEX additional boosting stations", total_by_year(capex_boost_year)),
        ("OPEX onshore pipe", total_by_year(opex_pipe_on_year)),
        ("OPEX offshore pipe", total_by_year(opex_pipe_off_year)),
        ("OPEX initial boosting stations", total_by_year(opex_init_year)),
        ("OPEX additional boosting stations", total_by_year(opex_boost_year)),
        ("Electricity cost initial boosting stations", total_by_year(el_init_year)),
        ("Electricity cost additional boosting stations", total_by_year(el_boost_year)),
        ("Shipping cost", total_by_year(ship_cost_year)),
        ("Penalty for uncaptured emissions", total_by_year(penalty_uncap_year)),
        ("CO2 sales revenue", total_by_year(sale_revenue_year)),
    ]

    recs = []
    for name, dic in rows:
        rec = {"Concept": name}
        rec.update(dic)
        rec["Total"] = sum(dic.values())
        recs.append(rec)

    totals = {yr: sum(r[yr] for r in recs) for yr in years}
    totals["Concept"] = "TOTAL"
    totals["Total"] = sum(totals[yr] for yr in years)
    recs.append(totals)

    cols = ["Concept"] + years + ["Total"]
    return pd.DataFrame(recs, columns=cols)


# ---------------------------------------------------------------------------
# 6) NETWORK LENGTH PER SCENARIO
# ---------------------------------------------------------------------------

def create_cost_length_per_scenario(m, years=None) -> pd.DataFrame:
    """
    Creates a DataFrame with network length (km) per scenario (LUS, EUS and HUS)
    in WIDE format:
        Scenario | 2030 | 2035 | 2040 | ... 
    Values are cumulative network length up to each time step.
    """

    if years is None:
        years = _years_from_T(m)

    # Sort time steps to ensure correct cumulative behavior
    T_list = sorted(list(m.T), key=lambda x: int(x))

    records = []
    for w in m.W:
        suf = _SCEN_SUFFIX.get(str(w), str(w))

        row = {"Scenario": suf}

        for i, t in enumerate(T_list):
            # Cumulative: pipelines installed in any tt <= t
            T_upto = T_list[: i + 1]

            total_length = 0.0

            # Onshore
            for p in m.P_on:
                installed_upto_t = any(value(_z_on(m, p, tt, w)) > 0.5 for tt in T_upto)
                if installed_upto_t:
                    total_length += value(m.L_on[p])

            # Offshore
            for p in m.P_off:
                installed_upto_t = any(value(_z_off(m, p, tt, w)) > 0.5 for tt in T_upto)
                if installed_upto_t:
                    total_length += value(m.L_off[p])

            # Column name = year (prefer 'years' list if aligned, else int(t))
            yr = years[i] if i < len(years) else int(t)
            row[int(yr)] = total_length

        records.append(row)

    # Build DataFrame and order columns: Scenario first, then years
    df = pd.DataFrame.from_records(records)

    year_cols = [int(y) for y in years]
    cols = ["Scenario"] + [c for c in year_cols if c in df.columns]

    return df[cols]


# ---------------------------------------------------------------------------
# 7) EXPORT: 4 sheets per scenario + Pipeline longitude per scenario
# ---------------------------------------------------------------------------

_SCEN_SUFFIX = {
    "low_utilization": "LUS",
    "base_utilization": "EUS",
    "high_utilization": "HUS",
}


def export_results(m, year_nodes: int | None = None, years=None):
    """
    Export results to a single Excel workbook:
      - Pipes summary per scenario
      - Nodes summary per scenario
      - Sinks capacity evolution per scenario
      - Utilization evolution per scenario
      - Cost breakdown per scenario
      - Cost breakdown in expected value (EV)
    """
    if years is None:
        years = _years_from_T(m)

    # Robust one-liner (if THETA does not exist -> 'UNK')
    theta_txt = f"{value(m.THETA):.2f}" if hasattr(m, "THETA") else "UNK"

    outpath = _OUTDIR / f"stochastic_results_theta_{theta_txt}.xlsx"
    with pd.ExcelWriter(outpath, engine="openpyxl", mode="w") as writer:
        for w in m.W:
            suf = _SCEN_SUFFIX.get(str(w), str(w))

            df_pipes = create_pipe_summary(m, w, years_flow=years)
            df_nodes = create_node_summary(m, w, year_nodes)
            df_sinks = create_sink_evolution(m, w, years_flow=years)
            df_util = create_util_evolution(m, w, years_flow=years)
            df_costs = create_cost_breakdown(m, w, years=years)

            df_pipes.to_excel(writer, sheet_name=f"{suf} - Pipes", index=False)
            _format_header_row(writer.sheets[f"{suf} - Pipes"])

            df_sinks.to_excel(writer, sheet_name=f"{suf} - Sinks capacity evolution", index=False)
            _format_header_row(writer.sheets[f"{suf} - Sinks capacity evolution"])

            df_util.to_excel(writer, sheet_name=f"{suf} - Utilization evolution", index=False)
            _format_header_row(writer.sheets[f"{suf} - Utilization evolution"])

            df_costs.to_excel(writer, sheet_name=f"{suf} - Cost breakdown", index=False)
            _format_header_row(writer.sheets[f"{suf} - Cost breakdown"])

        # Network length per scenario (pipe longitude)
        df_ntwrk_len = create_cost_length_per_scenario(m, years=years)
        df_ntwrk_len.to_excel(writer, sheet_name=f"Network length per scenario", index=False)
        _format_header_row(writer.sheets[f"Network length per scenario"])

    print(f"✔ Excel written to {outpath.resolve()}")
