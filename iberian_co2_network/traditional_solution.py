# traditional_solution.py

"""
Utility functions to extract results from the Pyomo model and export them to Excel
spreadsheets (scenario-aware).

All monetary figures are assumed to already be in present-value terms (M€).
"""

from pathlib import Path
from typing import Dict, Any, Sequence

import math
import pandas as pd
from pyomo.environ import value
from openpyxl.styles import Font, Alignment, Border, Side

# ---------------------------------------------------------------------------
# Output folder and global settings
# ---------------------------------------------------------------------------

_OUTDIR = Path("output_traditional_model")
_OUTDIR.mkdir(exist_ok=True)

years_per_step = 5 # number of years per time step in the model
KM_PER_BOOST_DEFAULT = 150.0  # traditional spacing (km per booster)


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
# Traditional boosters: ex-post cost computation
# ---------------------------------------------------------------------------

def _n_boosters_from_length(length_km: float, km_per_boost: float) -> int:
    """0 if L ≤ km_per_boost; 1 if km_per_boost < L ≤ 2*km_per_boost; etc."""
    return max(0, math.ceil(length_km / km_per_boost) - 1)

def _traditional_booster_costs_per_year_per_scenario(
    m,
    w,
    km_per_boost: float = KM_PER_BOOST_DEFAULT,
    multiply_opex_by_years: bool = False,
) -> Dict[str, Dict[Any, float]]:
    """
    Compute per-year CAPEX/OPEX/Electricity for 'traditional' boosters in scenario w.
    - One booster every `km_per_boost` km along any built pipeline segment.
    - CAPEX is paid once at the build year (delta of act_*).
    - OPEX per active year per booster.
    - Electricity proportional to flow and number of boosters.
    Returns dicts { 'capex':{t:...}, 'opex':{t:...}, 'el':{t:...} } keyed by model's t.
    """
    T_list = list(m.T.data())
    t0 = T_list[0]
    t_prev = {t: (None if i == 0 else T_list[i-1]) for i, t in enumerate(T_list)}

    # Precompute boosters per pipe (on/off)
    nboost_on  = {p: _n_boosters_from_length(value(m.L_on[p]),  km_per_boost) for p in m.P_on}
    nboost_off = {p: _n_boosters_from_length(value(m.L_off[p]), km_per_boost) for p in m.P_off}

    # Params by year
    cins_boost = {t: float(value(m.cins_boost[t])) for t in m.T}  # M€ / booster
    cop_boost  = {t: float(value(m.cop_boost [t])) for t in m.T}  # M€ / booster / step
    cel_boost  = {t: float(value(m.cel_boost [t])) for t in m.T}  # M€ / (Mt) per booster

    capex_year = {t: 0.0 for t in m.T}
    opex_year  = {t: 0.0 for t in m.T}
    el_year    = {t: 0.0 for t in m.T}

    opex_mult = years_per_step if multiply_opex_by_years else 1

    for t in T_list:
        # --- CAPEX: only at build year (delta act)
        for p in m.P_on:
            act_t    = value(m.act_on[p, t, w])
            act_prev = 0.0 if t == t0 else value(m.act_on[p, t_prev[t], w])
            delta    = max(0.0, act_t - act_prev)
            capex_year[t] += cins_boost[t] * nboost_on[p] * delta
        for p in m.P_off:
            act_t    = value(m.act_off[p, t, w])
            act_prev = 0.0 if t == t0 else value(m.act_off[p, t_prev[t], w])
            delta    = max(0.0, act_t - act_prev)
            capex_year[t] += cins_boost[t] * nboost_off[p] * delta

        # --- OPEX: boosters active in year t
        on_active  = sum(nboost_on [p] * value(m.act_on [p, t, w]) for p in m.P_on)
        off_active = sum(nboost_off[p] * value(m.act_off[p, t, w]) for p in m.P_off)
        opex_year[t] += cop_boost[t] * (on_active + off_active) * opex_mult

        # --- Electricity: all flow crosses each booster in series
        flow_on  = sum(nboost_on [p] * value(m.q_on [p, t, w]) for p in m.P_on)
        flow_off = sum(nboost_off[p] * value(m.q_off[p, t, w]) for p in m.P_off)
        el_year[t] += cel_boost[t] * (flow_on + flow_off)

    return {"capex": capex_year, "opex": opex_year, "el": el_year}

def _traditional_booster_costs_EV(
    m,
    km_per_boost: float = KM_PER_BOOST_DEFAULT,
    multiply_opex_by_years: bool = False,
) -> Dict[str, Dict[Any, float] | float]:
    """
    Expected-value (over scenarios) traditional booster costs by year and totals.
    Returns {
      'by_year': { 'capex': {t:...}, 'opex':{t:...}, 'el':{t:...} },
      'expected_totals': {'capex':x, 'opex':y, 'el':z, 'total':x+y+z}
    }
    """
    # Init per-year EV dicts
    capex_ev = {t: 0.0 for t in m.T}
    opex_ev  = {t: 0.0 for t in m.T}
    el_ev    = {t: 0.0 for t in m.T}

    for w in m.W:
        pw = value(m.prob[w])
        per_w = _traditional_booster_costs_per_year_per_scenario(
            m, w, km_per_boost=km_per_boost, multiply_opex_by_years=multiply_opex_by_years
        )
        for t in m.T:
            capex_ev[t] += pw * per_w["capex"][t]
            opex_ev [t] += pw * per_w["opex"][t]
            el_ev   [t] += pw * per_w["el"][t]

    capex_tot = sum(capex_ev[t] for t in m.T)
    opex_tot  = sum(opex_ev [t] for t in m.T)
    el_tot    = sum(el_ev   [t] for t in m.T)

    return {
        "by_year": {"capex": capex_ev, "opex": opex_ev, "el": el_ev},
        "expected_totals": {
            "capex": capex_tot, "opex": opex_tot, "el": el_tot, "total": capex_tot + opex_tot + el_tot
        }
    }


# ---------------------------------------------------------------------------
# 1) PIPE SUMMARY TABLE (per scenario)
# ---------------------------------------------------------------------------

def create_pipe_summary(m, w, years_flow=None, km_per_boost: float = KM_PER_BOOST_DEFAULT):
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

    records = []

    for p in list(P_on) + list(P_off):
        is_on = p in P_on

        # Lengths
        L = value(m.L_on[p]) if is_on else value(m.L_off[p])
        Lh = None

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

        # Traditional boosters (purely from length)
        n_boost_trad = _n_boosters_from_length(L, km_per_boost) if installed else None

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
            "Traditional boosters ({} km rule)".format(int(km_per_boost)): n_boost_trad,
            "Installation Year": year if installed else None,
            "Present Value Cost [M€]": capex if installed else None,
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
# 5) COST BREAKDOWN TABLE (per scenario) — includes TRADITIONAL boosters
# ---------------------------------------------------------------------------

def create_cost_breakdown(m, w: str, years=None, km_per_boost: float = KM_PER_BOOST_DEFAULT):
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

    # -------------------------------------------------------------------
    # Traditional boosters (ex-post): CAPEX/OPEX/Elec per year (scenario w)
    #   - 1 booster every KM_PER_BOOST_DEFAULT km (rule already in helper)
    #   - CAPEX paid only at installation year (z == 1)
    #   - OPEX paid every time step for all boosters that are active that step
    #   - Electricity proportional to flow and number of boosters in series
    # -------------------------------------------------------------------

    def _pipe_capacity_on(p, t, w):
        """Max capacity (Mt per time step) implied by chosen diameter for an onshore pipe."""
        return sum(value(m.qmax[d]) * value(_b_diam_on(m, p, d, w)) for d in m.D)

    def _pipe_capacity_off(p, t, w):
        """Max capacity (Mt per time step) implied by chosen diameter for an offshore pipe."""
        return sum(value(m.qmax[d]) * value(_b_diam_off(m, p, d, w)) for d in m.D)

    def capex_boost_year(yr):
        """
        CAPEX boosters in year yr:
          sum_over_pipes( n_boosters(p) * cins_boost[yr] * cap_max(pipe) )  only if built in yr.
        """
        t = _t(yr)
        if t is None:
            return 0.0

        total = 0.0

        # Onshore pipes
        for p in m.P_on:
            z = value(_z_on(m, p, t, w))
            if z <= 0.5:
                continue  # not built this step

            L = value(m.L_on[p])
            nboost = _n_boosters_from_length(L, KM_PER_BOOST_DEFAULT)
            if nboost <= 0:
                continue

            cap = _pipe_capacity_on(p, t, w)
            total += value(m.cins_boost[t]) * cap * nboost

        # Offshore pipes
        for p in m.P_off:
            z = value(_z_off(m, p, t, w))
            if z <= 0.5:
                continue  # not built this step

            L = value(m.L_off[p])
            nboost = _n_boosters_from_length(L, KM_PER_BOOST_DEFAULT)
            if nboost <= 0:
                continue

            cap = _pipe_capacity_off(p, t, w)
            total += value(m.cins_boost[t]) * cap * nboost

        return total

    def opex_boost_year(yr):
        """
        OPEX boosters in year yr:
          pay for all boosters that exist and are active in yr (i.e., pipes active in yr),
          using the OPEX unit of yr.
        """
        t = _t(yr)
        if t is None:
            return 0.0

        total = 0.0

        # Onshore pipes
        for p in m.P_on:
            act = value(m.act_on[p, t, w])
            if act <= 1e-12:
                continue  # pipe not active => no booster OPEX

            L = value(m.L_on[p])
            nboost = _n_boosters_from_length(L, KM_PER_BOOST_DEFAULT)
            if nboost <= 0:
                continue

            cap = _pipe_capacity_on(p, t, w)
            total += value(m.cop_boost[t]) * cap * nboost * act

        # Offshore pipes
        for p in m.P_off:
            act = value(m.act_off[p, t, w])
            if act <= 1e-12:
                continue

            L = value(m.L_off[p])
            nboost = _n_boosters_from_length(L, KM_PER_BOOST_DEFAULT)
            if nboost <= 0:
                continue

            cap = _pipe_capacity_off(p, t, w)
            total += value(m.cop_boost[t]) * cap * nboost * act

        return total

    def el_boost_year(yr):
        """
        Electricity cost in year yr:
          (total flow through boosters, counting flow multiple times if multiple boosters in series)
          * cel_boost[yr]
        Does NOT depend on pipe diameter/capacity.
        """
        t = _t(yr)
        if t is None:
            return 0.0

        flow_through_boosters = 0.0

        # Onshore
        for p in m.P_on:
            act = value(m.act_on[p, t, w])
            if act <= 1e-12:
                continue

            L = value(m.L_on[p])
            nboost = _n_boosters_from_length(L, KM_PER_BOOST_DEFAULT)
            if nboost <= 0:
                continue

            q = value(m.q_on[p, t, w])
            flow_through_boosters += nboost * q  # counted once per booster

        # Offshore
        for p in m.P_off:
            act = value(m.act_off[p, t, w])
            if act <= 1e-12:
                continue

            L = value(m.L_off[p])
            nboost = _n_boosters_from_length(L, KM_PER_BOOST_DEFAULT)
            if nboost <= 0:
                continue

            q = value(m.q_off[p, t, w])
            flow_through_boosters += nboost * q

        return value(m.cel_boost[t]) * flow_through_boosters




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

    outpath = _OUTDIR / f"stochastic_traditional_results_theta_{theta_txt}.xlsx"
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
