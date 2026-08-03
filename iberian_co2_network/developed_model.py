# developed_model.py

from pyomo import environ as pyo


def build_model(data: dict) -> pyo.ConcreteModel:
    """
    Create and return a Pyomo ConcreteModel given a fully-populated `data` dict.

    Notes:
    - Time steps are coarse (e.g., 2030, 2035, 2040, 2045, 2050), each representing a 5-year block.
    - P1 variables are scenario-independent (no omega index).
    - P2 variables are scenario-dependent (omega index), with non-anticipativity enforced at the first time step.
    - Pipe diameter is fixed per pipeline (no time index). Once a pipeline is built, its diameter does not change.
    """
    m = pyo.ConcreteModel(name="Iberian CCS network")

    # Definition of the capture and utilization target and tolerance
    THETA = 0.1
    TOL_P = 0.02  # ±2%
    m.THETA = pyo.Param(initialize=THETA)

    # ------------------------------------------------------------------
    # 0) Sets
    # ------------------------------------------------------------------
    m.E = pyo.Set(initialize=data["E"])
    m.S = pyo.Set(initialize=data["S"])
    m.A = pyo.Set(initialize=data["A"])
    m.M = pyo.Set(initialize=data["M"])
    m.K = pyo.Set(initialize=data["K"])
    m.N = pyo.Set(initialize=data["N"])

    m.W = pyo.Set(initialize=data["scenarios"])  # omega

    # Pipeline sets (P1: scenario-independent; P2: scenario-dependent)
    m.P1_on = pyo.Set(initialize=data["P1_on"])   # first stage (no endpoint in K)
    m.P1_off = pyo.Set(initialize=data["P1_off"])
    m.P2_on = pyo.Set(initialize=data["P2_on"])   # second stage (>= 1 endpoint in K)
    m.P2_off = pyo.Set(initialize=data["P2_off"])

    m.P_on = pyo.Set(initialize=data["P_on"])
    m.P_off = pyo.Set(initialize=data["P_off"])

    # Combined pipe set
    m.P = pyo.Set(initialize=list(data["P_on"]) + list(data["P_off"]))

    m.D = pyo.Set(initialize=data["D"])
    m.T = pyo.Set(initialize=data["T"], ordered=True)

    # Helpers: use data dicts directly
    IN = data["IN"]
    OUT = data["OUT"]

    # ------------------------------------------------------------------
    # 1) Decision variables
    # ------------------------------------------------------------------

    # Scenario-dependent flows and states
    m.q_on = pyo.Var(m.P_on, m.T, m.W, domain=pyo.NonNegativeReals)    # q_{p,t,w}
    m.q_off = pyo.Var(m.P_off, m.T, m.W, domain=pyo.NonNegativeReals)  # q_{p,t,w}
    m.q_dom = pyo.Var(m.N, m.T, m.W, domain=pyo.NonNegativeReals)      # dominant inflow q*_{i,t,w}

    m.qcap = pyo.Var(m.E, m.T, m.W, domain=pyo.NonNegativeReals)
    m.qimp = pyo.Var(m.M, m.T, m.W, domain=pyo.NonNegativeReals)
    m.qexp = pyo.Var(m.M, m.T, m.W, domain=pyo.NonNegativeReals)
    m.qstore = pyo.Var(m.S, m.T, m.W, domain=pyo.NonNegativeReals)
    m.qsale = pyo.Var(m.K, m.T, m.W, domain=pyo.NonNegativeReals)

    m.alpha = pyo.Var(m.S, m.T, m.W, domain=pyo.NonNegativeReals)

    m.pi_node = pyo.Var(m.N, m.T, m.W, domain=pyo.NonNegativeReals)
    m.pi_orig = pyo.Var(m.P, m.T, m.W, domain=pyo.NonNegativeReals)
    m.pi_dest = pyo.Var(m.P, m.T, m.W, domain=pyo.NonNegativeReals)

    # Dominance and shipping (scenario-dependent)
    m.y_on = pyo.Var(m.P_on, m.T, m.W, domain=pyo.Binary)
    m.y_off = pyo.Var(m.P_off, m.T, m.W, domain=pyo.Binary)
    m.b_ship = pyo.Var(m.P_off, m.T, m.W, domain=pyo.Binary)
    m.nship = pyo.Var(m.P_off, m.T, m.W, domain=pyo.NonNegativeIntegers)

    # Linearization auxiliaries for booster costs (scenario-dependent because they multiply flow)
    m.w_boost_on1 = pyo.Var(m.P_on, m.T, m.W, domain=pyo.NonNegativeReals)
    m.w_boost_on2 = pyo.Var(m.P_on, m.T, m.W, domain=pyo.NonNegativeReals)
    m.w_boost_off = pyo.Var(m.P_off, m.T, m.W, domain=pyo.NonNegativeReals)

    # Auxiliary variables (= 1 if, in scenario w and time step t, pipeline p is active with diameter d)
    m.u_on = pyo.Var(m.P_on, m.D, m.T, m.W, domain=pyo.UnitInterval)   # continuous in [0, 1]
    m.u_off = pyo.Var(m.P_off, m.D, m.T, m.W, domain=pyo.UnitInterval)

    # First-stage binaries (P1): scenario-independent
    m.b_diam_on_P1 = pyo.Var(m.P1_on, m.D, domain=pyo.Binary)
    m.b_diam_off_P1 = pyo.Var(m.P1_off, m.D, domain=pyo.Binary)

    m.z_on_P1 = pyo.Var(m.P1_on, m.T, domain=pyo.Binary)   # one-shot build in a time step
    m.z_off_P1 = pyo.Var(m.P1_off, m.T, domain=pyo.Binary)

    m.brep_on1_P1 = pyo.Var(m.P1_on, m.T, domain=pyo.Binary)
    m.brep_on2_P1 = pyo.Var(m.P1_on, m.T, domain=pyo.Binary)
    m.brep_off_P1 = pyo.Var(m.P1_off, m.T, domain=pyo.Binary)

    # Second-stage binaries (P2): scenario-dependent
    m.b_diam_on_P2 = pyo.Var(m.P2_on, m.D, m.W, domain=pyo.Binary)
    m.b_diam_off_P2 = pyo.Var(m.P2_off, m.D, m.W, domain=pyo.Binary)

    m.z_on_P2 = pyo.Var(m.P2_on, m.T, m.W, domain=pyo.Binary)
    m.z_off_P2 = pyo.Var(m.P2_off, m.T, m.W, domain=pyo.Binary)

    m.brep_on1_P2 = pyo.Var(m.P2_on, m.T, m.W, domain=pyo.Binary)
    m.brep_on2_P2 = pyo.Var(m.P2_on, m.T, m.W, domain=pyo.Binary)
    m.brep_off_P2 = pyo.Var(m.P2_off, m.T, m.W, domain=pyo.Binary)

    # Initial compression station decisions (scenario-indexed)
    m.z_init = pyo.Var(m.E, m.T, m.W, domain=pyo.Binary)

    # ------------------------------------------------------------------
    # 2) Parameters (imported from `data`)
    # ------------------------------------------------------------------

    # Emissions (g_{e,t})
    m.g = pyo.Param(m.E, m.T, initialize=data["emission"], within=pyo.NonNegativeReals)

    # Maximum storage capacity at sinks (Mt)
    m.cap_store = pyo.Param(m.S, initialize=data["store_cap"], within=pyo.NonNegativeReals)

    # Maximum utilization capacity (scenario-specific)
    m.g_cap = pyo.Param(m.K, m.T, m.W, initialize=data["g_cap"], within=pyo.NonNegativeReals)

    # Pipe length (km)
    m.L_on = pyo.Param(m.P_on, initialize={p: data["L"][p] for p in data["P_on"]}, within=pyo.NonNegativeReals)
    m.L_off = pyo.Param(m.P_off, initialize={p: data["L"][p] for p in data["P_off"]}, within=pyo.NonNegativeReals)

    # Distance to highest point (km)
    m.Lh_on = pyo.Param(m.P_on, initialize={p: data["Lh"][p] for p in data["P_on"]}, within=pyo.NonNegativeReals)
    m.Lh_off = pyo.Param(m.P_off, initialize={p: data["Lh"][p] for p in data["P_off"]}, within=pyo.NonNegativeReals)

    # Elevation pressure drops
    m.dP_elev_high = pyo.Param(m.P, initialize=data["dP_elev_high"], within=pyo.NonNegativeReals)
    m.dP_elev_far = pyo.Param(m.P, initialize=data["dP_elev_far"], within=pyo.Reals)

    # Friction pressure drops
    m.dP_frict_high = pyo.Param(m.P * m.D, initialize=data["dP_frict_high"], within=pyo.NonNegativeReals)
    m.dP_frict_far = pyo.Param(m.P * m.D, initialize=data["dP_frict_far"], within=pyo.NonNegativeReals)

    # Pipe capacity by diameter (t/year)
    m.qmax = pyo.Param(m.D, initialize=data["qmax"], within=pyo.NonNegativeReals)

    # Pipeline CAPEX (M€/km)
    m.cins_on = pyo.Param(m.D, m.T, initialize=data["cins_on"], within=pyo.NonNegativeReals)
    m.cins_off = pyo.Param(m.D, m.T, initialize=data["cins_off"], within=pyo.NonNegativeReals)

    # Pipeline OPEX (M€/km·yr)
    m.cop_on = pyo.Param(m.D, m.T, initialize=data["cop_on"], within=pyo.NonNegativeReals)
    m.cop_off = pyo.Param(m.D, m.T, initialize=data["cop_off"], within=pyo.NonNegativeReals)

    # Carbon allowance price (M€/MtCO2)
    m.allow_price = pyo.Param(m.T, initialize=data["allow_price"], within=pyo.NonNegativeReals)

    # Capture & injection costs (M€/MtCO2)
    m.capture_cost = pyo.Param(m.T, initialize=data["capture_cost"], within=pyo.NonNegativeReals)
    m.injection_cost = pyo.Param(m.T, initialize=data["injection_cost"], within=pyo.NonNegativeReals)

    # Shipping costs
    m.ship_fixed_cost = pyo.Param(m.T, initialize=data["ship_fixed_cost"], within=pyo.NonNegativeReals)  # (M€/cycle)
    m.ship_fuel_cost = pyo.Param(m.T, initialize=data["ship_fuel_cost"], within=pyo.NonNegativeReals)    # (M€/km/cycle)

    # Compression costs (M€/MtCO2)
    m.cins_init = pyo.Param(m.T, initialize=data["cins_init"], within=pyo.NonNegativeReals)
    m.cop_init = pyo.Param(m.T, initialize=data["cop_init"], within=pyo.NonNegativeReals)
    m.cel_init = pyo.Param(m.T, initialize=data["cel_init"], within=pyo.NonNegativeReals)

    m.cins_boost = pyo.Param(m.T, initialize=data["cins_boost"], within=pyo.NonNegativeReals)
    m.cop_boost = pyo.Param(m.T, initialize=data["cop_boost"], within=pyo.NonNegativeReals)
    m.cel_boost = pyo.Param(m.T, initialize=data["cel_boost"], within=pyo.NonNegativeReals)

    # Selling revenue to industry (M€/MtCO2)
    m.selling_revenue = pyo.Param(m.T, initialize=data["selling_revenue"], within=pyo.NonNegativeReals)

    # Deterministic targets (base)
    m.seq_target = pyo.Param(m.T, initialize=data["seq_target"], within=pyo.NonNegativeReals)
    m.util_target = pyo.Param(m.T, initialize=data["util_target"], within=pyo.NonNegativeReals)

    # Maximum number of incoming pipes (for dominant selection big-M)
    max_deg = max(len(IN[i]) for i in data["N"])

    # Constants (pressures, big-M values, ship capacity)
    m.p_emit = pyo.Param(initialize=data["p_emit"], within=pyo.NonNegativeReals)
    m.delta_p_boost = pyo.Param(initialize=data["delta_p_boost"], within=pyo.NonNegativeReals)
    m.p_min = pyo.Param(initialize=data["p_min"], within=pyo.NonNegativeReals)

    m.M_flow = pyo.Param(initialize=data["M_flow"], within=pyo.NonNegativeReals)
    m.M_press = pyo.Param(initialize=data["M_press"], within=pyo.NonNegativeReals)
    m.M_eur = pyo.Param(initialize=data["M_eur"], within=pyo.NonNegativeReals)
    m.M_deg = pyo.Param(initialize=max_deg, within=pyo.NonNegativeReals)

    m.ship_capacity = pyo.Param(initialize=data["ship_capacity"], within=pyo.NonNegativeReals)

    # Scenario probabilities
    m.prob = pyo.Param(m.W, initialize=data["scenario_prob"], within=pyo.UnitInterval)

    # Maximum emissions per source node used for design (t/year)
    max_emit = {e: max(data["emission"][e, tt] for tt in data["T"]) for e in data["E"]}
    m.qcap_design = pyo.Param(m.E, initialize=max_emit)

    # ------------------------------------------------------------------
    # 3) Additional static parameters (nodes & pipes)
    # ------------------------------------------------------------------
    m.height = pyo.Param(m.N, initialize=data["height"], within=pyo.NonNegativeReals)
    m.country_n = pyo.Param(m.N, initialize=data["country_n"], within=pyo.Any)
    m.stage_n = pyo.Param(m.N, initialize=data["stage_n"], within=pyo.Any)

    # Pipe endpoints
    m.start = pyo.Param(m.P, initialize=data["start"], within=m.N)
    m.end = pyo.Param(m.P, initialize=data["end"], within=m.N)

    # Simple penalties
    m.pen_city = pyo.Param(m.P, initialize=data["pen_city"], within=pyo.NonNegativeReals)
    m.pen_slope = pyo.Param(m.P, initialize=data["pen_slope"], within=pyo.NonNegativeReals)

    m.tmethod_p = pyo.Param(m.P, initialize=data["tmethod_p"], within=pyo.Any)
    m.stage_p = pyo.Param(m.P, initialize=data["stage_p"], within=pyo.Any)

    m.country_i = pyo.Param(m.P, initialize=data["country_i"], within=pyo.Any)
    m.country_j = pyo.Param(m.P, initialize=data["country_j"], within=pyo.Any)

    m.n1_type = pyo.Param(m.P, initialize=data["n1_type"], within=pyo.Any)
    m.n2_type = pyo.Param(m.P, initialize=data["n2_type"], within=pyo.Any)

    # ------------------------------------------------------------------
    # 4) Mass balance constraints (scenario-indexed)
    # ------------------------------------------------------------------

    # Global mass balance per year and scenario
    def global_mass_balance_rule(m, t, w):
        generation = sum(m.qcap[e, t, w] for e in m.E)
        storage = sum(m.qstore[s, t, w] for s in m.S)
        sale = sum(m.qsale[k, t, w] for k in m.K)
        return generation == storage + sale

    m.GlobalMassBalance = pyo.Constraint(m.T, m.W, rule=global_mass_balance_rule)

    # Mass balance at emitter nodes
    def emitter_mass_balance_rule(m, e, t, w):
        inflow = (
            sum(m.q_on[p, t, w] for p in IN[e] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in IN[e] if p in m.P_off)
        )
        outflow = (
            sum(m.q_on[p, t, w] for p in OUT[e] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in OUT[e] if p in m.P_off)
        )
        return inflow + m.qcap[e, t, w] == outflow

    m.EmitterMassBalance = pyo.Constraint(m.E, m.T, m.W, rule=emitter_mass_balance_rule)

    # Mass balance at sink nodes
    def sink_mass_balance_rule(m, s, t, w):
        inflow = (
            sum(m.q_on[p, t, w] for p in IN[s] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in IN[s] if p in m.P_off)
        )
        outflow = (
            sum(m.q_on[p, t, w] for p in OUT[s] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in OUT[s] if p in m.P_off)
        )
        return inflow == m.qstore[s, t, w] + outflow

    m.SinkMassBalance = pyo.Constraint(m.S, m.T, m.W, rule=sink_mass_balance_rule)

    # Mass balance at auxiliary nodes
    def aux_mass_balance_rule(m, a, t, w):
        inflow = (
            sum(m.q_on[p, t, w] for p in IN[a] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in IN[a] if p in m.P_off)
        )
        outflow = (
            sum(m.q_on[p, t, w] for p in OUT[a] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in OUT[a] if p in m.P_off)
        )
        return inflow == outflow

    m.AuxMassBalance = pyo.Constraint(m.A, m.T, m.W, rule=aux_mass_balance_rule)

    # Mass balance at trade nodes (kept although qimp/qexp are fixed to 0)
    def trade_mass_balance_rule(m, m_ctry, t, w):
        inflow = (
            sum(m.q_on[p, t, w] for p in IN[m_ctry] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in IN[m_ctry] if p in m.P_off)
        )
        outflow = (
            sum(m.q_on[p, t, w] for p in OUT[m_ctry] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in OUT[m_ctry] if p in m.P_off)
        )
        return inflow + m.qimp[m_ctry, t, w] == m.qexp[m_ctry, t, w] + outflow

    m.TradeMassBalance = pyo.Constraint(m.M, m.T, m.W, rule=trade_mass_balance_rule)

    # Mass balance at utilization nodes
    def util_mass_balance_rule(m, k, t, w):
        inflow = (
            sum(m.q_on[p, t, w] for p in IN[k] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in IN[k] if p in m.P_off)
        )
        outflow = (
            sum(m.q_on[p, t, w] for p in OUT[k] if p in m.P_on)
            + sum(m.q_off[p, t, w] for p in OUT[k] if p in m.P_off)
        )
        return inflow == m.qsale[k, t, w] + outflow

    m.UtilMassBalance = pyo.Constraint(m.K, m.T, m.W, rule=util_mass_balance_rule)

    # ------------------------------------------------------------------
    # 5) Sink storage evolution (scenario-indexed)
    # ------------------------------------------------------------------
    T_list = list(m.T.ordered_data())
    _prev_year = {t: None if i == 0 else T_list[i - 1] for i, t in enumerate(T_list)}

    def storage_evolution_rule(m, s, t, w):
        if t == T_list[0]:
            return pyo.Constraint.Skip
        t_prev = _prev_year[t]
        return m.alpha[s, t, w] == m.alpha[s, t_prev, w] - m.qstore[s, t_prev, w]

    m.StorageEvolution = pyo.Constraint(m.S, m.T, m.W, rule=storage_evolution_rule)

    def storage_initial_transition_rule(m, s, w):
        t0 = T_list[0]
        t1 = T_list[1]
        return m.alpha[s, t1, w] == m.cap_store[s] - m.qstore[s, t0, w]

    m.InitialStorageTransition = pyo.Constraint(m.S, m.W, rule=storage_initial_transition_rule)

    def injection_limit_rule(m, s, t, w):
        return m.qstore[s, t, w] <= m.alpha[s, t, w]

    m.InjectionLimitedByStorage = pyo.Constraint(m.S, m.T, m.W, rule=injection_limit_rule)

    def injection_limit_first_year_rule(m, s, w):
        t0 = T_list[0]
        return m.qstore[s, t0, w] <= m.cap_store[s]

    m.InjectionLimitFirstYear = pyo.Constraint(m.S, m.W, rule=injection_limit_first_year_rule)

    def storage_initial_level_rule(m, s, w):
        t0 = T_list[0]
        return m.alpha[s, t0, w] == m.cap_store[s]

    m.StorageInitialLevel = pyo.Constraint(m.S, m.W, rule=storage_initial_level_rule)

    # ------------------------------------------------------------------
    # 6) Helpers for P1/P2 effective variables
    # ------------------------------------------------------------------
    def b_diam_on_eff(m, p, d, w):
        return m.b_diam_on_P1[p, d] if p in m.P1_on else m.b_diam_on_P2[p, d, w]

    def b_diam_off_eff(m, p, d, w):
        return m.b_diam_off_P1[p, d] if p in m.P1_off else m.b_diam_off_P2[p, d, w]

    def z_on_eff(m, p, t, w):
        return m.z_on_P1[p, t] if p in m.P1_on else m.z_on_P2[p, t, w]

    def z_off_eff(m, p, t, w):
        return m.z_off_P1[p, t] if p in m.P1_off else m.z_off_P2[p, t, w]

    def brep_on1_eff(m, p, t, w):
        return m.brep_on1_P1[p, t] if p in m.P1_on else m.brep_on1_P2[p, t, w]

    def brep_on2_eff(m, p, t, w):
        return m.brep_on2_P1[p, t] if p in m.P1_on else m.brep_on2_P2[p, t, w]

    def brep_off_eff(m, p, t, w):
        return m.brep_off_P1[p, t] if p in m.P1_off else m.brep_off_P2[p, t, w]

    # ------------------------------------------------------------------
    # 7) Activation expressions for pipelines (scenario-indexed because P2 depends on w)
    # ------------------------------------------------------------------
    def _act_on_rule(m, p, t, w):
        return sum(z_on_eff(m, p, tau, w) for tau in m.T if tau <= t)

    m.act_on = pyo.Expression(m.P_on, m.T, m.W, rule=_act_on_rule)

    def _act_off_rule(m, p, t, w):
        return sum(z_off_eff(m, p, tau, w) for tau in m.T if tau <= t)

    m.act_off = pyo.Expression(m.P_off, m.T, m.W, rule=_act_off_rule)

    # ------------------------------------------------------------------
    # 8) Diameter selection constraints
    # ------------------------------------------------------------------

    # Flow capacity by chosen diameter
    def onshore_flow_capacity_rule(m, p, t, w):
        return m.q_on[p, t, w] <= sum(m.qmax[d] * b_diam_on_eff(m, p, d, w) for d in m.D)

    m.OnshoreFlowCapacity = pyo.Constraint(m.P_on, m.T, m.W, rule=onshore_flow_capacity_rule)

    def offshore_flow_capacity_rule(m, p, t, w):
        return m.q_off[p, t, w] <= sum(m.qmax[d] * b_diam_off_eff(m, p, d, w) for d in m.D)

    m.OffshoreFlowCapacity = pyo.Constraint(m.P_off, m.T, m.W, rule=offshore_flow_capacity_rule)

    # Single-diameter per pipeline
    def onshore_single_diam_P1(m, p):
        return sum(m.b_diam_on_P1[p, d] for d in m.D) <= 1

    m.OnshoreSingleDiameter_P1 = pyo.Constraint(m.P1_on, rule=onshore_single_diam_P1)

    def onshore_single_diam_P2(m, p, w):
        return sum(m.b_diam_on_P2[p, d, w] for d in m.D) <= 1

    m.OnshoreSingleDiameter_P2 = pyo.Constraint(m.P2_on, m.W, rule=onshore_single_diam_P2)

    def offshore_single_diam_P1(m, p):
        return sum(m.b_diam_off_P1[p, d] for d in m.D) <= 1

    m.OffshoreSingleDiameter_P1 = pyo.Constraint(m.P1_off, rule=offshore_single_diam_P1)

    def offshore_single_diam_P2(m, p, w):
        return sum(m.b_diam_off_P2[p, d, w] for d in m.D) <= 1

    m.OffshoreSingleDiameter_P2 = pyo.Constraint(m.P2_off, m.W, rule=offshore_single_diam_P2)

    # Flow allowed only after the pipeline is built (or shipped for offshore)
    def onshore_flow_after_build_rule(m, p, t, w):
        return m.q_on[p, t, w] <= m.M_flow * m.act_on[p, t, w]

    m.OnshoreFlowAfterBuild = pyo.Constraint(m.P_on, m.T, m.W, rule=onshore_flow_after_build_rule)

    def offshore_flow_after_build_rule(m, p, t, w):
        return m.q_off[p, t, w] <= m.M_flow * (m.act_off[p, t, w] + m.b_ship[p, t, w])

    m.OffshoreFlowAfterBuild = pyo.Constraint(m.P_off, m.T, m.W, rule=offshore_flow_after_build_rule)

    # Diameter only if the segment is built (P1 and P2)
    def diam_only_if_built_on_P1(m, p):
        return sum(m.b_diam_on_P1[p, d] for d in m.D) <= sum(m.z_on_P1[p, t] for t in m.T)

    m.DiamRequiresBuildOn_P1 = pyo.Constraint(m.P1_on, rule=diam_only_if_built_on_P1)

    def diam_only_if_built_off_P1(m, p):
        return sum(m.b_diam_off_P1[p, d] for d in m.D) <= sum(m.z_off_P1[p, t] for t in m.T)

    m.DiamRequiresBuildOff_P1 = pyo.Constraint(m.P1_off, rule=diam_only_if_built_off_P1)

    def diam_only_if_built_on_P2(m, p, w):
        return sum(m.b_diam_on_P2[p, d, w] for d in m.D) <= sum(m.z_on_P2[p, t, w] for t in m.T)

    m.DiamRequiresBuildOn_P2 = pyo.Constraint(m.P2_on, m.W, rule=diam_only_if_built_on_P2)

    def diam_only_if_built_off_P2(m, p, w):
        return sum(m.b_diam_off_P2[p, d, w] for d in m.D) <= sum(m.z_off_P2[p, t, w] for t in m.T)

    m.DiamRequiresBuildOff_P2 = pyo.Constraint(m.P2_off, m.W, rule=diam_only_if_built_off_P2)

    # Construction requires a diameter choice (P1 and P2)
    def onshore_constr_needs_diam_P1(m, p, t):
        return m.z_on_P1[p, t] <= sum(m.b_diam_on_P1[p, d] for d in m.D)

    m.OnshoreConstrNeedsDiam_P1 = pyo.Constraint(m.P1_on, m.T, rule=onshore_constr_needs_diam_P1)

    def offshore_constr_needs_diam_P1(m, p, t):
        return m.z_off_P1[p, t] <= sum(m.b_diam_off_P1[p, d] for d in m.D)

    m.OffshoreConstrNeedsDiam_P1 = pyo.Constraint(m.P1_off, m.T, rule=offshore_constr_needs_diam_P1)

    def onshore_constr_needs_diam_P2(m, p, t, w):
        return m.z_on_P2[p, t, w] <= sum(m.b_diam_on_P2[p, d, w] for d in m.D)

    m.OnshoreConstrNeedsDiam_P2 = pyo.Constraint(m.P2_on, m.T, m.W, rule=onshore_constr_needs_diam_P2)

    def offshore_constr_needs_diam_P2(m, p, t, w):
        return m.z_off_P2[p, t, w] <= sum(m.b_diam_off_P2[p, d, w] for d in m.D)

    m.OffshoreConstrNeedsDiam_P2 = pyo.Constraint(m.P2_off, m.T, m.W, rule=offshore_constr_needs_diam_P2)

    # ------------------------------------------------------------------
    # 9) Pressure constraints (scenario-indexed)
    # ------------------------------------------------------------------

    # Fixed pressure at every emitter node
    def fixed_emitter_pressure_rule(m, e, t, w):
        return m.pi_node[e, t, w] == m.p_emit

    m.FixedEmitterPressure = pyo.Constraint(m.E, m.T, m.W, rule=fixed_emitter_pressure_rule)

    # Shorthands
    Mpi = m.M_press
    Dpi = m.delta_p_boost
    pmin = m.p_min

    # Onshore pressure drop (far point)
    def onshore_pressure_low_rule(m, p, t, w):
        frict = sum(b_diam_on_eff(m, p, d, w) * m.dP_frict_far[p, d] for d in m.D)
        nboost = brep_on1_eff(m, p, t, w) + brep_on2_eff(m, p, t, w)
        return (
            m.pi_dest[p, t, w]
            >= m.pi_orig[p, t, w] - m.dP_elev_far[p] - frict + Dpi * nboost - Mpi * (1 - m.act_on[p, t, w])
        )

    def onshore_pressure_up_rule(m, p, t, w):
        frict = sum(b_diam_on_eff(m, p, d, w) * m.dP_frict_far[p, d] for d in m.D)
        nboost = brep_on1_eff(m, p, t, w) + brep_on2_eff(m, p, t, w)
        return (
            m.pi_dest[p, t, w]
            <= m.pi_orig[p, t, w] - m.dP_elev_far[p] - frict + Dpi * nboost + Mpi * (1 - m.act_on[p, t, w])
        )

    m.PressDropOnshore_LB = pyo.Constraint(m.P_on, m.T, m.W, rule=onshore_pressure_low_rule)
    m.PressDropOnshore_UB = pyo.Constraint(m.P_on, m.T, m.W, rule=onshore_pressure_up_rule)

    # Offshore pressure drop (far point)
    def offshore_pressure_low_rule(m, p, t, w):
        frict = sum(b_diam_off_eff(m, p, d, w) * m.dP_frict_far[p, d] for d in m.D)
        return (
            m.pi_dest[p, t, w]
            >= m.pi_orig[p, t, w] - frict + Dpi * brep_off_eff(m, p, t, w) - Mpi * (1 - m.act_off[p, t, w])
        )

    def offshore_pressure_up_rule(m, p, t, w):
        frict = sum(b_diam_off_eff(m, p, d, w) * m.dP_frict_far[p, d] for d in m.D)
        return (
            m.pi_dest[p, t, w]
            <= m.pi_orig[p, t, w] - frict + Dpi * brep_off_eff(m, p, t, w) + Mpi * (1 - m.act_off[p, t, w])
        )

    m.PressDropOffshore_LB = pyo.Constraint(m.P_off, m.T, m.W, rule=offshore_pressure_low_rule)
    m.PressDropOffshore_UB = pyo.Constraint(m.P_off, m.T, m.W, rule=offshore_pressure_up_rule)

    # Minimum pressure at the highest point (onshore)
    def onshore_high_point_rule(m, p, t, w):
        frict = sum(b_diam_on_eff(m, p, d, w) * m.dP_frict_high[p, d] for d in m.D)
        nboost = brep_on1_eff(m, p, t, w) + brep_on2_eff(m, p, t, w)
        return (
            m.pi_orig[p, t, w]
            >= pmin + m.dP_elev_high[p] + frict - Dpi * nboost - Mpi * (1 - m.act_on[p, t, w])
        )

    m.HighPointMinPressure = pyo.Constraint(m.P_on, m.T, m.W, rule=onshore_high_point_rule)

    # Minimum pressure at pipe outlet
    m.DestMinPressureOn = pyo.Constraint(m.P_on, m.T, m.W, rule=lambda m, p, t, w: m.pi_dest[p, t, w] >= pmin)
    m.DestMinPressureOff = pyo.Constraint(m.P_off, m.T, m.W, rule=lambda m, p, t, w: m.pi_dest[p, t, w] >= pmin)

    # Minimum node pressure
    m.MinNodePressure = pyo.Constraint(m.N, m.T, m.W, rule=lambda m, i, t, w: m.pi_node[i, t, w] >= pmin)

    # Link origin node pressure to pipeline origin
    def origin_pressure_link_rule(m, p, t, w):
        return m.pi_orig[p, t, w] == m.pi_node[m.start[p], t, w]

    m.OriginPressureLink = pyo.Constraint(m.P, m.T, m.W, rule=origin_pressure_link_rule)

    # ------------------------------------------------------------------
    # 10) Dominant-pipe selection and node pressure (scenario-indexed)
    # ------------------------------------------------------------------

    def _in_present_rule(m, i, t, w):
        # Number of incoming active links (including ship as a possible incoming link)
        return (
            sum(m.act_on[p, t, w] for p in IN[i] if p in m.P_on)
            + sum(m.act_off[p, t, w] for p in IN[i] if p in m.P_off)
            + sum(m.b_ship[p, t, w] for p in IN[i] if p in m.P_off)
        )

    m.in_present = pyo.Expression(m.N, m.T, m.W, rule=_in_present_rule)

    # At most one dominant incoming pipe
    def dom_le_one_rule(m, i, t, w):
        n_dom = sum(m.y_on[p, t, w] for p in IN[i] if p in m.P_on) + sum(
            m.y_off[p, t, w] for p in IN[i] if p in m.P_off
        )
        return n_dom <= 1

    m.DominantAtMostOne = pyo.Constraint(m.N, m.T, m.W, rule=dom_le_one_rule)

    # At least one dominant when there is any incoming active link
    def dom_ge_one_if_active_rule(m, i, t, w):
        n_dom = sum(m.y_on[p, t, w] for p in IN[i] if p in m.P_on) + sum(
            m.y_off[p, t, w] for p in IN[i] if p in m.P_off
        )
        return n_dom >= m.in_present[i, t, w] / m.M_deg

    m.DominantIfActive = pyo.Constraint(m.N, m.T, m.W, rule=dom_ge_one_if_active_rule)

    # A pipe can be dominant only if it is active
    m.DominantActiveOn = pyo.Constraint(
        m.P_on, m.T, m.W, rule=lambda m, p, t, w: m.y_on[p, t, w] <= m.act_on[p, t, w]
    )
    m.DominantActiveOff = pyo.Constraint(
        m.P_off, m.T, m.W, rule=lambda m, p, t, w: m.y_off[p, t, w] <= m.act_off[p, t, w]
    )

    # Big-M linking q_dom with the selected dominant pipe flow
    Mq = m.M_flow
    m.DominantFlowUB_on = pyo.ConstraintList()
    m.DominantFlowLB_on = pyo.ConstraintList()
    m.DominantFlowUB_off = pyo.ConstraintList()
    m.DominantFlowLB_off = pyo.ConstraintList()

    for i in m.N:
        for t in m.T:
            for w in m.W:
                for p in IN[i]:
                    if p in m.P_on:
                        m.DominantFlowUB_on.add(m.q_dom[i, t, w] <= m.q_on[p, t, w] + Mq * (1 - m.y_on[p, t, w]))
                        m.DominantFlowLB_on.add(m.q_dom[i, t, w] >= m.q_on[p, t, w] - Mq * (1 - m.y_on[p, t, w]))
                    elif p in m.P_off:
                        m.DominantFlowUB_off.add(m.q_dom[i, t, w] <= m.q_off[p, t, w] + Mq * (1 - m.y_off[p, t, w]))
                        m.DominantFlowLB_off.add(m.q_dom[i, t, w] >= m.q_off[p, t, w] - Mq * (1 - m.y_off[p, t, w]))
                    else:
                        continue

    # Non-dominant pipes must not exceed the dominant flow
    m.NonDominantFlow = pyo.ConstraintList()
    for i in m.N:
        for t in m.T:
            for w in m.W:
                for p in IN[i]:
                    if p in m.P_on:
                        m.NonDominantFlow.add(m.q_on[p, t, w] <= m.q_dom[i, t, w] + Mq * (1 - m.y_on[p, t, w]))
                    elif p in m.P_off:
                        m.NonDominantFlow.add(m.q_off[p, t, w] <= m.q_dom[i, t, w] + Mq * (1 - m.y_off[p, t, w]))
                    else:
                        continue

    # Node pressure equals the destination pressure of the dominant incoming pipe (big-M)
    m.NodePressureEq = pyo.ConstraintList()
    for i in m.N:
        for t in m.T:
            for w in m.W:
                for p in IN[i]:
                    if p in m.P_on:
                        m.NodePressureEq.add(m.pi_node[i, t, w] >= m.pi_dest[p, t, w] - Mpi * (1 - m.y_on[p, t, w]))
                        m.NodePressureEq.add(m.pi_node[i, t, w] <= m.pi_dest[p, t, w] + Mpi * (1 - m.y_on[p, t, w]))
                    elif p in m.P_off:
                        m.NodePressureEq.add(m.pi_node[i, t, w] >= m.pi_dest[p, t, w] - Mpi * (1 - m.y_off[p, t, w]))
                        m.NodePressureEq.add(m.pi_node[i, t, w] <= m.pi_dest[p, t, w] + Mpi * (1 - m.y_off[p, t, w]))
                    else:
                        continue

    # Keep non-dominant pipes close to node pressure (tolerance band)
    Delta = 30  # bar

    def node_pressure_consistency_lb(m, i, p, t, w):
        if p not in IN[i]:
            return pyo.Constraint.Skip
        if p in m.P_on:
            active = m.act_on[p, t, w]
        elif p in m.P_off:
            active = m.act_off[p, t, w] + m.b_ship[p, t, w]
        else:
            return pyo.Constraint.Skip
        return m.pi_node[i, t, w] >= m.pi_dest[p, t, w] - Delta - m.M_press * (1 - active)

    def node_pressure_consistency_ub(m, i, p, t, w):
        if p not in IN[i]:
            return pyo.Constraint.Skip
        if p in m.P_on:
            active = m.act_on[p, t, w]
        elif p in m.P_off:
            active = m.act_off[p, t, w] + m.b_ship[p, t, w]
        else:
            return pyo.Constraint.Skip
        return m.pi_node[i, t, w] <= m.pi_dest[p, t, w] + Delta + m.M_press * (1 - active)

    m.NodePressTol_LB = pyo.Constraint(m.N, m.P, m.T, m.W, rule=node_pressure_consistency_lb)
    m.NodePressTol_UB = pyo.Constraint(m.N, m.P, m.T, m.W, rule=node_pressure_consistency_ub)

    # ------------------------------------------------------------------
    # 11) Booster installation rules (activation and sequence)
    # ------------------------------------------------------------------

    # Booster can exist only if the pipe is active (P1: scenario-independent build decisions)
    def booster_onshore_activation_P1(m, p, t):
        act = sum(m.z_on_P1[p, tau] for tau in m.T if tau <= t)
        return m.brep_on1_P1[p, t] + m.brep_on2_P1[p, t] <= act

    m.BoosterOnshoreActivation_P1 = pyo.Constraint(m.P1_on, m.T, rule=booster_onshore_activation_P1)

    def booster_offshore_activation_P1(m, p, t):
        act = sum(m.z_off_P1[p, tau] for tau in m.T if tau <= t)
        return m.brep_off_P1[p, t] <= act

    m.BoosterOffshoreActivation_P1 = pyo.Constraint(m.P1_off, m.T, rule=booster_offshore_activation_P1)

    # P2: per scenario (use act_on/act_off expressions)
    def booster_onshore_activation_P2(m, p, t, w):
        return brep_on1_eff(m, p, t, w) + brep_on2_eff(m, p, t, w) <= m.act_on[p, t, w]

    m.BoosterOnshoreActivation_P2 = pyo.Constraint(m.P2_on, m.T, m.W, rule=booster_onshore_activation_P2)

    def booster_offshore_activation_P2(m, p, t, w):
        return brep_off_eff(m, p, t, w) <= m.act_off[p, t, w]

    m.BoosterOffshoreActivation_P2 = pyo.Constraint(m.P2_off, m.T, m.W, rule=booster_offshore_activation_P2)

    # Second onshore booster only if the first is used
    m.BoosterOnshoreSequence_P1 = pyo.Constraint(
        m.P1_on, m.T, rule=lambda m, p, t: m.brep_on2_P1[p, t] <= m.brep_on1_P1[p, t]
    )
    m.BoosterOnshoreSequence_P2 = pyo.Constraint(
        m.P2_on, m.T, m.W, rule=lambda m, p, t, w: m.brep_on2_P2[p, t, w] <= m.brep_on1_P2[p, t, w]
    )

    # ------------------------------------------------------------------
    # 12) Ship transport constraints (scenario-indexed)
    # ------------------------------------------------------------------

    # Mutually exclusive: offshore pipeline or ship
    def ship_pipeline_exclusivity_rule(m, p, t, w):
        return m.act_off[p, t, w] + m.b_ship[p, t, w] <= 1

    m.ShipPipelineExclusive = pyo.Constraint(m.P_off, m.T, m.W, rule=ship_pipeline_exclusivity_rule)

    gv = m.ship_capacity
    Mship = m.M_flow

    # Flow = nship * gv when shipping is selected (big-M relaxation otherwise)
    def ship_flow_upper_rule(m, p, t, w):
        return m.q_off[p, t, w] - gv * m.nship[p, t, w] <= Mship * (1 - m.b_ship[p, t, w])

    def ship_flow_lower_rule(m, p, t, w):
        return gv * m.nship[p, t, w] - m.q_off[p, t, w] <= Mship * (1 - m.b_ship[p, t, w])

    m.ShipFlowUB = pyo.Constraint(m.P_off, m.T, m.W, rule=ship_flow_upper_rule)
    m.ShipFlowLB = pyo.Constraint(m.P_off, m.T, m.W, rule=ship_flow_lower_rule)

    # Destination pressure equals emitter pressure when shipping
    def ship_dest_pressure_lower_rule(m, p, t, w):
        return m.pi_dest[p, t, w] >= m.p_emit - Mpi * (1 - m.b_ship[p, t, w])

    def ship_dest_pressure_upper_rule(m, p, t, w):
        return m.pi_dest[p, t, w] <= m.p_emit + Mpi * (1 - m.b_ship[p, t, w])

    m.ShipDestPressureLB = pyo.Constraint(m.P_off, m.T, m.W, rule=ship_dest_pressure_lower_rule)
    m.ShipDestPressureUB = pyo.Constraint(m.P_off, m.T, m.W, rule=ship_dest_pressure_upper_rule)

    # ------------------------------------------------------------------
    # 13) Utilization capacity constraint (scenario-specific capacity)
    # ------------------------------------------------------------------
    m.UtilizationCapacity = pyo.Constraint(
        m.K, m.T, m.W, rule=lambda m, k, t, w: m.qsale[k, t, w] <= m.g_cap[k, t, w]
    )

    # ------------------------------------------------------------------
    # 14) Capture limit constraint (deterministic emissions)
    # ------------------------------------------------------------------
    m.CaptureLimit = pyo.Constraint(m.E, m.T, m.W, rule=lambda m, e, t, w: m.qcap[e, t, w] <= m.g[e, t])

    # ------------------------------------------------------------------
    # 15) Capture and utilization target constraints (scenario-specific targets, ± tolerance)
    # ------------------------------------------------------------------

    # Scenario multipliers for the utilization target
    def _util_mult_init(m, w):
        mapping = {
            "low_utilization": 0.5,
            "base_utilization": 1.0,
            "high_utilization": 1.5,
        }
        return mapping[w]

    m.util_mult = pyo.Param(m.W, initialize=_util_mult_init, within=pyo.PositiveReals)

    # Total target (seq + util) stays constant across scenarios
    def _tot_target_init(m, t):
        return pyo.value(m.seq_target[t] + m.util_target[t])

    m.tot_target = pyo.Param(m.T, initialize=_tot_target_init)

    # Scenario-specific utilization target
    def _util_target_scen_init(m, w, t):
        return pyo.value(m.util_mult[w] * m.util_target[t])

    m.util_target_scen = pyo.Param(m.W, m.T, initialize=_util_target_scen_init)

    # Scenario-specific sequestration target: total - util
    def _seq_target_scen_init(m, w, t):
        val = pyo.value(m.tot_target[t] - m.util_target_scen[w, t])
        return max(0.0, val)

    m.seq_target_scen = pyo.Param(m.W, m.T, initialize=_seq_target_scen_init)

    # Capture target per scenario (stored + utilized)
    def capture_target_lb_per_scen(m, t, w):
        target = m.THETA * m.tot_target[t]
        total_capture_w = sum(m.qcap[e, t, w] for e in m.E)
        return total_capture_w >= (1 - TOL_P) * target

    def capture_target_ub_per_scen(m, t, w):
        target = m.THETA * m.tot_target[t]
        total_capture_w = sum(m.qcap[e, t, w] for e in m.E)
        return total_capture_w <= (1 + TOL_P) * target

    m.CaptureTargetPerScen_LB = pyo.Constraint(m.T, m.W, rule=capture_target_lb_per_scen)
    m.CaptureTargetPerScen_UB = pyo.Constraint(m.T, m.W, rule=capture_target_ub_per_scen)

    # Utilization target per scenario
    def util_target_lb_per_scen(m, t, w):
        target = m.THETA * m.util_target_scen[w, t]
        total_util_w = sum(m.qsale[k, t, w] for k in m.K)
        return total_util_w >= (1 - TOL_P) * target

    def util_target_ub_per_scen(m, t, w):
        target = m.THETA * m.util_target_scen[w, t]
        total_util_w = sum(m.qsale[k, t, w] for k in m.K)
        return total_util_w <= (1 + TOL_P) * target

    m.UtilTargetPerScen_LB = pyo.Constraint(m.T, m.W, rule=util_target_lb_per_scen)
    m.UtilTargetPerScen_UB = pyo.Constraint(m.T, m.W, rule=util_target_ub_per_scen)

    # Storage (sequestration) target per scenario
    def storage_target_lb_per_scen(m, t, w):
        target = m.THETA * m.seq_target_scen[w, t]
        total_store_w = sum(m.qstore[s, t, w] for s in m.S)
        return total_store_w >= (1 - TOL_P) * target

    def storage_target_ub_per_scen(m, t, w):
        target = m.THETA * m.seq_target_scen[w, t]
        total_store_w = sum(m.qstore[s, t, w] for s in m.S)
        return total_store_w <= (1 + TOL_P) * target

    m.StorageTargetPerScen_LB = pyo.Constraint(m.T, m.W, rule=storage_target_lb_per_scen)
    m.StorageTargetPerScen_UB = pyo.Constraint(m.T, m.W, rule=storage_target_ub_per_scen)

    # ------------------------------------------------------------------
    # 16) Monotonicity and one-shot construction
    # ------------------------------------------------------------------
    t0 = T_list[0]

    # Booster monotonicity (P1): once installed, it remains installed
    m.BoosterMonotOn1_P1 = pyo.ConstraintList()
    m.BoosterMonotOn2_P1 = pyo.ConstraintList()
    m.BoosterMonotOff_P1 = pyo.ConstraintList()
    for t_prev, t_next in zip(T_list[:-1], T_list[1:]):
        for p in m.P1_on:
            m.BoosterMonotOn1_P1.add(m.brep_on1_P1[p, t_next] >= m.brep_on1_P1[p, t_prev])
            m.BoosterMonotOn2_P1.add(m.brep_on2_P1[p, t_next] >= m.brep_on2_P1[p, t_prev])
        for p in m.P1_off:
            m.BoosterMonotOff_P1.add(m.brep_off_P1[p, t_next] >= m.brep_off_P1[p, t_prev])

    # Booster monotonicity (P2): per scenario
    m.BoosterMonotOn1_P2 = pyo.ConstraintList()
    m.BoosterMonotOn2_P2 = pyo.ConstraintList()
    m.BoosterMonotOff_P2 = pyo.ConstraintList()
    for t_prev, t_next in zip(T_list[:-1], T_list[1:]):
        for w in m.W:
            for p in m.P2_on:
                m.BoosterMonotOn1_P2.add(m.brep_on1_P2[p, t_next, w] >= m.brep_on1_P2[p, t_prev, w])
                m.BoosterMonotOn2_P2.add(m.brep_on2_P2[p, t_next, w] >= m.brep_on2_P2[p, t_prev, w])
            for p in m.P2_off:
                m.BoosterMonotOff_P2.add(m.brep_off_P2[p, t_next, w] >= m.brep_off_P2[p, t_prev, w])

    # One-shot construction per pipeline segment
    m.SingleBuildOn_P1 = pyo.Constraint(m.P1_on, rule=lambda m, p: sum(m.z_on_P1[p, t] for t in m.T) <= 1)
    m.SingleBuildOff_P1 = pyo.Constraint(m.P1_off, rule=lambda m, p: sum(m.z_off_P1[p, t] for t in m.T) <= 1)

    m.SingleBuildOn_P2 = pyo.Constraint(
        m.P2_on, m.W, rule=lambda m, p, w: sum(m.z_on_P2[p, t, w] for t in m.T) <= 1
    )
    m.SingleBuildOff_P2 = pyo.Constraint(
        m.P2_off, m.W, rule=lambda m, p, w: sum(m.z_off_P2[p, t, w] for t in m.T) <= 1
    )

    # ------------------------------------------------------------------
    # 17) Initial compression stations and capture activation
    # ------------------------------------------------------------------
    m.SingleInit = pyo.Constraint(m.E, m.W, rule=lambda m, e, w: sum(m.z_init[e, t, w] for t in m.T) <= 1)

    def capture_requires_init_rule(m, e, t, w):
        return m.qcap[e, t, w] <= m.M_flow * sum(m.z_init[e, tau, w] for tau in m.T if tau <= t)

    m.CaptureNeedsInit = pyo.Constraint(m.E, m.T, m.W, rule=capture_requires_init_rule)

    # ------------------------------------------------------------------
    # 18) Non-anticipativity constraints at the first time step t0
    #     (two-stage: uncertainty is revealed after t0)
    # ------------------------------------------------------------------
    W_list = list(m.W.ordered_data())
    w0 = W_list[0]

    # Non-anticipativity for z_init at t0
    m.NA_z_init = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for e in m.E:
            m.NA_z_init.add(m.z_init[e, t0, w] == m.z_init[e, t0, w0])

    # A) P2 binaries: build and boosters equal at t0
    m.NA_z_on_P2 = pyo.ConstraintList()
    m.NA_z_off_P2 = pyo.ConstraintList()
    m.NA_brep_on1 = pyo.ConstraintList()
    m.NA_brep_on2 = pyo.ConstraintList()
    m.NA_brep_off = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for p in m.P2_on:
            m.NA_z_on_P2.add(m.z_on_P2[p, t0, w] == m.z_on_P2[p, t0, w0])
            m.NA_brep_on1.add(m.brep_on1_P2[p, t0, w] == m.brep_on1_P2[p, t0, w0])
            m.NA_brep_on2.add(m.brep_on2_P2[p, t0, w] == m.brep_on2_P2[p, t0, w0])
        for p in m.P2_off:
            m.NA_z_off_P2.add(m.z_off_P2[p, t0, w] == m.z_off_P2[p, t0, w0])
            m.NA_brep_off.add(m.brep_off_P2[p, t0, w] == m.brep_off_P2[p, t0, w0])

    # B) Conditional non-anticipativity for P2 diameters:
    #    Enforce equality across scenarios only if built at t0 (in either scenario)
    m.NA_bdiam_on_P2_cond = pyo.ConstraintList()
    m.NA_bdiam_off_P2_cond = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for p in m.P2_on:
            for d in m.D:
                m.NA_bdiam_on_P2_cond.add(
                    m.b_diam_on_P2[p, d, w] - m.b_diam_on_P2[p, d, w0] <= 1 - m.z_on_P2[p, t0, w]
                )
                m.NA_bdiam_on_P2_cond.add(
                    m.b_diam_on_P2[p, d, w0] - m.b_diam_on_P2[p, d, w] <= 1 - m.z_on_P2[p, t0, w]
                )
                m.NA_bdiam_on_P2_cond.add(
                    m.b_diam_on_P2[p, d, w] - m.b_diam_on_P2[p, d, w0] <= 1 - m.z_on_P2[p, t0, w0]
                )
                m.NA_bdiam_on_P2_cond.add(
                    m.b_diam_on_P2[p, d, w0] - m.b_diam_on_P2[p, d, w] <= 1 - m.z_on_P2[p, t0, w0]
                )

        for p in m.P2_off:
            for d in m.D:
                m.NA_bdiam_off_P2_cond.add(
                    m.b_diam_off_P2[p, d, w] - m.b_diam_off_P2[p, d, w0] <= 1 - m.z_off_P2[p, t0, w]
                )
                m.NA_bdiam_off_P2_cond.add(
                    m.b_diam_off_P2[p, d, w0] - m.b_diam_off_P2[p, d, w] <= 1 - m.z_off_P2[p, t0, w]
                )
                m.NA_bdiam_off_P2_cond.add(
                    m.b_diam_off_P2[p, d, w] - m.b_diam_off_P2[p, d, w0] <= 1 - m.z_off_P2[p, t0, w0]
                )
                m.NA_bdiam_off_P2_cond.add(
                    m.b_diam_off_P2[p, d, w0] - m.b_diam_off_P2[p, d, w] <= 1 - m.z_off_P2[p, t0, w0]
                )

    # C) Operational variables at t0 equal across scenarios
    m.NA_b_ship = pyo.ConstraintList()
    m.NA_nship = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for p in m.P_off:
            m.NA_b_ship.add(m.b_ship[p, t0, w] == m.b_ship[p, t0, w0])
            m.NA_nship.add(m.nship[p, t0, w] == m.nship[p, t0, w0])

    m.NA_q_on = pyo.ConstraintList()
    m.NA_q_off = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for p in m.P_on:
            m.NA_q_on.add(m.q_on[p, t0, w] == m.q_on[p, t0, w0])
        for p in m.P_off:
            m.NA_q_off.add(m.q_off[p, t0, w] == m.q_off[p, t0, w0])

    m.NA_pi_node = pyo.ConstraintList()
    m.NA_pi_orig = pyo.ConstraintList()
    m.NA_pi_dest = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for i in m.N:
            m.NA_pi_node.add(m.pi_node[i, t0, w] == m.pi_node[i, t0, w0])
        for p in m.P:
            m.NA_pi_orig.add(m.pi_orig[p, t0, w] == m.pi_orig[p, t0, w0])
            m.NA_pi_dest.add(m.pi_dest[p, t0, w] == m.pi_dest[p, t0, w0])

    m.NA_y_on = pyo.ConstraintList()
    m.NA_y_off = pyo.ConstraintList()
    m.NA_qdom = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for p in m.P_on:
            m.NA_y_on.add(m.y_on[p, t0, w] == m.y_on[p, t0, w0])
        for p in m.P_off:
            m.NA_y_off.add(m.y_off[p, t0, w] == m.y_off[p, t0, w0])
        for i in m.N:
            m.NA_qdom.add(m.q_dom[i, t0, w] == m.q_dom[i, t0, w0])

    m.NA_qcap = pyo.ConstraintList()
    m.NA_qstore = pyo.ConstraintList()
    m.NA_qsale = pyo.ConstraintList()
    for w in m.W:
        if w == w0:
            continue
        for e in m.E:
            m.NA_qcap.add(m.qcap[e, t0, w] == m.qcap[e, t0, w0])
        for s in m.S:
            m.NA_qstore.add(m.qstore[s, t0, w] == m.qstore[s, t0, w0])
        for k in m.K:
            m.NA_qsale.add(m.qsale[k, t0, w] == m.qsale[k, t0, w0])

    # ------------------------------------------------------------------
    # 19) Linearization of booster-flow products: w_boost_* = q * b_booster_*
    # ------------------------------------------------------------------
    Uq = m.M_flow

    # Onshore: booster 1
    m.WOn1UB1 = pyo.Constraint(
        m.P_on, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_on1[p, t, w] <= Uq * brep_on1_eff(m, p, t, w)
    )
    m.WOn1UB2 = pyo.Constraint(
        m.P_on, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_on1[p, t, w] <= m.q_on[p, t, w]
    )
    m.WOn1LB = pyo.Constraint(
        m.P_on, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_on1[p, t, w] >= m.q_on[p, t, w] - Uq * (1 - brep_on1_eff(m, p, t, w))
    )

    # Onshore: booster 2
    m.WOn2UB1 = pyo.Constraint(
        m.P_on, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_on2[p, t, w] <= Uq * brep_on2_eff(m, p, t, w)
    )
    m.WOn2UB2 = pyo.Constraint(
        m.P_on, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_on2[p, t, w] <= m.q_on[p, t, w]
    )
    m.WOn2LB = pyo.Constraint(
        m.P_on, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_on2[p, t, w] >= m.q_on[p, t, w] - Uq * (1 - brep_on2_eff(m, p, t, w))
    )

    # Offshore: single booster
    m.WOffUB1 = pyo.Constraint(
        m.P_off, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_off[p, t, w] <= Uq * brep_off_eff(m, p, t, w)
    )
    m.WOffUB2 = pyo.Constraint(
        m.P_off, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_off[p, t, w] <= m.q_off[p, t, w]
    )
    m.WOffLB = pyo.Constraint(
        m.P_off, m.T, m.W,
        rule=lambda m, p, t, w: m.w_boost_off[p, t, w] >= m.q_off[p, t, w] - Uq * (1 - brep_off_eff(m, p, t, w))
    )

    # ------------------------------------------------------------------
    # 20) Linearization for the objective function: u_on/off = act_on/off * b_diam_on/off
    # ------------------------------------------------------------------
    def u_on_ub1(m, p, d, t, w):
        return m.u_on[p, d, t, w] <= m.act_on[p, t, w]

    def u_on_ub2(m, p, d, t, w):
        return m.u_on[p, d, t, w] <= b_diam_on_eff(m, p, d, w)

    def u_on_lb(m, p, d, t, w):
        return m.u_on[p, d, t, w] >= m.act_on[p, t, w] + b_diam_on_eff(m, p, d, w) - 1

    m.UOnUB1 = pyo.Constraint(m.P_on, m.D, m.T, m.W, rule=u_on_ub1)
    m.UOnUB2 = pyo.Constraint(m.P_on, m.D, m.T, m.W, rule=u_on_ub2)
    m.UOnLB  = pyo.Constraint(m.P_on, m.D, m.T, m.W, rule=u_on_lb)

    def u_off_ub1(m, p, d, t, w):
        return m.u_off[p, d, t, w] <= m.act_off[p, t, w]

    def u_off_ub2(m, p, d, t, w):
        return m.u_off[p, d, t, w] <= b_diam_off_eff(m, p, d, w)

    def u_off_lb(m, p, d, t, w):
        return m.u_off[p, d, t, w] >= m.act_off[p, t, w] + b_diam_off_eff(m, p, d, w) - 1

    m.UOffUB1 = pyo.Constraint(m.P_off, m.D, m.T, m.W, rule=u_off_ub1)
    m.UOffUB2 = pyo.Constraint(m.P_off, m.D, m.T, m.W, rule=u_off_ub2)
    m.UOffLB  = pyo.Constraint(m.P_off, m.D, m.T, m.W, rule=u_off_lb)

    # ------------------------------------------------------------------
    # 21) Costs & objective (expected total cost over the full horizon)
    # ------------------------------------------------------------------
    years_per_step = 5

    # Active flag for initial stations (scenario-dependent because z_init is indexed by w)
    m.act_init = pyo.Expression(
        m.E, m.T, m.W,
        rule=lambda m, e, t, w: sum(m.z_init[e, tau, w] for tau in m.T if tau <= t)
    )

    # Big-M values for CAPEX linearization
    capex_max_on = max(
        data["cins_on"][d, t]
        * max(data["L"][p] for p in data["P_on"])
        * max(data["pen_city"][p] * data["pen_slope"][p] for p in data["P_on"])
        for d in data["D"] for t in data["T"]
    ) + 1e-3

    capex_max_off = max(
        data["cins_off"][d, t]
        * max(data["L"][p] for p in data["P_off"])
        * max(data["pen_city"][p] * data["pen_slope"][p] for p in data["P_off"])
        for d in data["D"] for t in data["T"]
    ) + 1e-3

    # CAPEX variables: P1 scenario-independent, P2 scenario-dependent
    m.c_pipe_on_P1 = pyo.Var(m.P1_on, m.T, domain=pyo.NonNegativeReals)
    m.c_pipe_off_P1 = pyo.Var(m.P1_off, m.T, domain=pyo.NonNegativeReals)
    m.c_pipe_on_P2 = pyo.Var(m.P2_on, m.T, m.W, domain=pyo.NonNegativeReals)
    m.c_pipe_off_P2 = pyo.Var(m.P2_off, m.T, m.W, domain=pyo.NonNegativeReals)

    # P1 onshore CAPEX (paid only in the build year)
    def cap_on_P1_ub(m, p, t):
        expr = sum(
            m.cins_on[d, t] * m.L_on[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_on_P1[p, d]
            for d in m.D
        )
        return m.c_pipe_on_P1[p, t] <= expr

    def cap_on_P1_lb(m, p, t):
        expr = sum(
            m.cins_on[d, t] * m.L_on[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_on_P1[p, d]
            for d in m.D
        )
        return m.c_pipe_on_P1[p, t] >= expr - capex_max_on * (1 - m.z_on_P1[p, t])

    def cap_on_P1_z(m, p, t):
        return m.c_pipe_on_P1[p, t] <= capex_max_on * m.z_on_P1[p, t]

    m.CapOnP1UB = pyo.Constraint(m.P1_on, m.T, rule=cap_on_P1_ub)
    m.CapOnP1LB = pyo.Constraint(m.P1_on, m.T, rule=cap_on_P1_lb)
    m.CapOnP1Z  = pyo.Constraint(m.P1_on, m.T, rule=cap_on_P1_z)

    # P1 offshore CAPEX (paid only in the build year)
    def cap_off_P1_ub(m, p, t):
        expr = sum(
            m.cins_off[d, t] * m.L_off[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_off_P1[p, d]
            for d in m.D
        )
        return m.c_pipe_off_P1[p, t] <= expr

    def cap_off_P1_lb(m, p, t):
        expr = sum(
            m.cins_off[d, t] * m.L_off[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_off_P1[p, d]
            for d in m.D
        )
        return m.c_pipe_off_P1[p, t] >= expr - capex_max_off * (1 - m.z_off_P1[p, t])

    def cap_off_P1_z(m, p, t):
        return m.c_pipe_off_P1[p, t] <= capex_max_off * m.z_off_P1[p, t]

    m.CapOffP1UB = pyo.Constraint(m.P1_off, m.T, rule=cap_off_P1_ub)
    m.CapOffP1LB = pyo.Constraint(m.P1_off, m.T, rule=cap_off_P1_lb)
    m.CapOffP1Z  = pyo.Constraint(m.P1_off, m.T, rule=cap_off_P1_z)

    # P2 onshore CAPEX (paid only in the build year)
    def cap_on_P2_ub(m, p, t, w):
        expr = sum(
            m.cins_on[d, t] * m.L_on[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_on_P2[p, d, w]
            for d in m.D
        )
        return m.c_pipe_on_P2[p, t, w] <= expr

    def cap_on_P2_lb(m, p, t, w):
        expr = sum(
            m.cins_on[d, t] * m.L_on[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_on_P2[p, d, w]
            for d in m.D
        )
        return m.c_pipe_on_P2[p, t, w] >= expr - capex_max_on * (1 - m.z_on_P2[p, t, w])

    def cap_on_P2_z(m, p, t, w):
        return m.c_pipe_on_P2[p, t, w] <= capex_max_on * m.z_on_P2[p, t, w]

    m.CapOnP2UB = pyo.Constraint(m.P2_on, m.T, m.W, rule=cap_on_P2_ub)
    m.CapOnP2LB = pyo.Constraint(m.P2_on, m.T, m.W, rule=cap_on_P2_lb)
    m.CapOnP2Z  = pyo.Constraint(m.P2_on, m.T, m.W, rule=cap_on_P2_z)

    # P2 offshore CAPEX (paid only in the build year)
    def cap_off_P2_ub(m, p, t, w):
        expr = sum(
            m.cins_off[d, t] * m.L_off[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_off_P2[p, d, w]
            for d in m.D
        )
        return m.c_pipe_off_P2[p, t, w] <= expr

    def cap_off_P2_lb(m, p, t, w):
        expr = sum(
            m.cins_off[d, t] * m.L_off[p] * m.pen_city[p] * m.pen_slope[p] * m.b_diam_off_P2[p, d, w]
            for d in m.D
        )
        return m.c_pipe_off_P2[p, t, w] >= expr - capex_max_off * (1 - m.z_off_P2[p, t, w])

    def cap_off_P2_z(m, p, t, w):
        return m.c_pipe_off_P2[p, t, w] <= capex_max_off * m.z_off_P2[p, t, w]

    m.CapOffP2UB = pyo.Constraint(m.P2_off, m.T, m.W, rule=cap_off_P2_ub)
    m.CapOffP2LB = pyo.Constraint(m.P2_off, m.T, m.W, rule=cap_off_P2_lb)
    m.CapOffP2Z  = pyo.Constraint(m.P2_off, m.T, m.W, rule=cap_off_P2_z)

    # Pipeline CAPEX: P1 counted once (no omega), P2 in expectation
    capex_pipe_on = sum(m.c_pipe_on_P1[p, t] for p in m.P1_on for t in m.T) + sum(
        m.prob[w] * sum(m.c_pipe_on_P2[p, t, w] for p in m.P2_on for t in m.T) for w in m.W
    )
    capex_pipe_off = sum(m.c_pipe_off_P1[p, t] for p in m.P1_off for t in m.T) + sum(
        m.prob[w] * sum(m.c_pipe_off_P2[p, t, w] for p in m.P2_off for t in m.T) for w in m.W
    )

    # Pipeline OPEX (expected), scaled by years per step
    opex_pipe_on = years_per_step * sum(
        m.prob[w] * sum(
            m.cop_on[d, t] * m.L_on[p] * m.pen_city[p] * m.pen_slope[p] * m.u_on[p, d, t, w]
            for p in m.P_on for d in m.D for t in m.T
        )
        for w in m.W
    )

    opex_pipe_off = years_per_step * sum(
        m.prob[w] * sum(
            m.cop_off[d, t] * m.L_off[p] * m.pen_city[p] * m.pen_slope[p] * m.u_off[p, d, t, w]
            for p in m.P_off for d in m.D for t in m.T
        )
        for w in m.W
    )

    # Shipping cost (expected)
    ship_cost = sum(
        m.prob[w] * sum(
            m.nship[p, t, w] * (m.ship_fixed_cost[t] + m.ship_fuel_cost[t] * m.L_off[p])
            for p in m.P_off for t in m.T
        )
        for w in m.W
    )

    # Capture and injection costs (expected)
    capture_cost = sum(
        m.prob[w] * sum(m.capture_cost[t] * m.qcap[e, t, w] for e in m.E for t in m.T)
        for w in m.W
    )
    injection_cost = sum(
        m.prob[w] * sum(m.injection_cost[t] * m.qstore[s, t, w] for s in m.S for t in m.T)
        for w in m.W
    )

    # Penalty for uncaptured emissions (expected)
    penalty_uncaptured = sum(
        m.prob[w] * sum(
            (sum(m.g[e, t] for e in m.E) - sum(m.qstore[s, t, w] for s in m.S) - sum(m.qsale[k, t, w] for k in m.K))
            * m.allow_price[t]
            for t in m.T
        )
        for w in m.W
    )

    # Initial compression stations (CAPEX, OPEX, and electricity) in expectation
    capex_init = sum(
        m.prob[w] * sum(m.cins_init[t] * m.qcap_design[e] * m.z_init[e, t, w] for e in m.E for t in m.T)
        for w in m.W
    )
    opex_init = sum(
        m.prob[w] * sum(m.cop_init[t] * m.qcap_design[e] * m.act_init[e, t, w] for e in m.E for t in m.T)
        for w in m.W
    )
    el_init = sum(
        m.prob[w] * sum(m.cel_init[t] * m.qcap[e, t, w] for e in m.E for t in m.T)
        for w in m.W
    )

    # Booster CAPEX variables (P1 scenario-independent; P2 scenario-dependent)
    m.c_boost_on1_P1 = pyo.Var(m.P1_on, m.T, domain=pyo.NonNegativeReals)
    m.c_boost_on2_P1 = pyo.Var(m.P1_on, m.T, domain=pyo.NonNegativeReals)
    m.c_boost_off_P1 = pyo.Var(m.P1_off, m.T, domain=pyo.NonNegativeReals)

    m.c_boost_on1_P2 = pyo.Var(m.P2_on, m.T, m.W, domain=pyo.NonNegativeReals)
    m.c_boost_on2_P2 = pyo.Var(m.P2_on, m.T, m.W, domain=pyo.NonNegativeReals)
    m.c_boost_off_P2 = pyo.Var(m.P2_off, m.T, m.W, domain=pyo.NonNegativeReals)

    # Booster OPEX variables (P1 scenario-independent; P2 scenario-dependent)
    m.w_boostcap_on1_P1 = pyo.Var(m.P1_on,  m.T,      domain=pyo.NonNegativeReals)
    m.w_boostcap_on2_P1 = pyo.Var(m.P1_on,  m.T,      domain=pyo.NonNegativeReals)
    m.w_boostcap_off_P1 = pyo.Var(m.P1_off, m.T,      domain=pyo.NonNegativeReals)

    m.w_boostcap_on1_P2 = pyo.Var(m.P2_on,  m.T, m.W, domain=pyo.NonNegativeReals)
    m.w_boostcap_on2_P2 = pyo.Var(m.P2_on,  m.T, m.W, domain=pyo.NonNegativeReals)
    m.w_boostcap_off_P2 = pyo.Var(m.P2_off, m.T, m.W, domain=pyo.NonNegativeReals)

    # Maximum pipeline capacity used for scaling booster-cost auxiliary variables
    QMAX_MAX = max(pyo.value(m.qmax[d]) for d in m.D)

    def pipe_cap_P1(m, p):
        # P1 diameter variables have no scenario index
        if p in m.P1_on:
            return sum(m.qmax[d] * m.b_diam_on_P1[p, d] for d in m.D)
        elif p in m.P1_off:
            return sum(m.qmax[d] * m.b_diam_off_P1[p, d] for d in m.D)
        return 0.0

    def pipe_cap_P2(m, p, w):
        if p in m.P2_on:
            return sum(m.qmax[d] * m.b_diam_on_P2[p, d, w] for d in m.D)
        elif p in m.P2_off:
            return sum(m.qmax[d] * m.b_diam_off_P2[p, d, w] for d in m.D)
        return 0.0

    # P1 booster OPEX linearization constraints
    def _boostcap_on1_P1_ub1(m, p, t):
        return m.w_boostcap_on1_P1[p, t] <= pipe_cap_P1(m, p)

    def _boostcap_on1_P1_ub2(m, p, t):
        return m.w_boostcap_on1_P1[p, t] <= QMAX_MAX * m.brep_on1_P1[p, t]

    def _boostcap_on1_P1_lb(m, p, t):
        return m.w_boostcap_on1_P1[p, t] >= pipe_cap_P1(m, p) - QMAX_MAX * (1 - m.brep_on1_P1[p, t])

    m.BoostCapOn1_P1_ub1 = pyo.Constraint(m.P1_on, m.T, rule=_boostcap_on1_P1_ub1)
    m.BoostCapOn1_P1_ub2 = pyo.Constraint(m.P1_on, m.T, rule=_boostcap_on1_P1_ub2)
    m.BoostCapOn1_P1_lb  = pyo.Constraint(m.P1_on, m.T, rule=_boostcap_on1_P1_lb)

    def _boostcap_on2_P1_ub1(m, p, t):
        return m.w_boostcap_on2_P1[p, t] <= pipe_cap_P1(m, p)

    def _boostcap_on2_P1_ub2(m, p, t):
        return m.w_boostcap_on2_P1[p, t] <= QMAX_MAX * m.brep_on2_P1[p, t]

    def _boostcap_on2_P1_lb(m, p, t):
        return m.w_boostcap_on2_P1[p, t] >= pipe_cap_P1(m, p) - QMAX_MAX * (1 - m.brep_on2_P1[p, t])

    m.BoostCapOn2_P1_ub1 = pyo.Constraint(m.P1_on, m.T, rule=_boostcap_on2_P1_ub1)
    m.BoostCapOn2_P1_ub2 = pyo.Constraint(m.P1_on, m.T, rule=_boostcap_on2_P1_ub2)
    m.BoostCapOn2_P1_lb  = pyo.Constraint(m.P1_on, m.T, rule=_boostcap_on2_P1_lb)

    def _boostcap_off_P1_ub1(m, p, t):
        return m.w_boostcap_off_P1[p, t] <= pipe_cap_P1(m, p)

    def _boostcap_off_P1_ub2(m, p, t):
        return m.w_boostcap_off_P1[p, t] <= QMAX_MAX * m.brep_off_P1[p, t]

    def _boostcap_off_P1_lb(m, p, t):
        return m.w_boostcap_off_P1[p, t] >= pipe_cap_P1(m, p) - QMAX_MAX * (1 - m.brep_off_P1[p, t])

    m.BoostCapOff_P1_ub1 = pyo.Constraint(m.P1_off, m.T, rule=_boostcap_off_P1_ub1)
    m.BoostCapOff_P1_ub2 = pyo.Constraint(m.P1_off, m.T, rule=_boostcap_off_P1_ub2)
    m.BoostCapOff_P1_lb  = pyo.Constraint(m.P1_off, m.T, rule=_boostcap_off_P1_lb)

    # P2 booster OPEX linearization constraints
    def _boostcap_on1_P2_ub1(m, p, t, w):
        return m.w_boostcap_on1_P2[p, t, w] <= pipe_cap_P2(m, p, w)

    def _boostcap_on1_P2_ub2(m, p, t, w):
        return m.w_boostcap_on1_P2[p, t, w] <= QMAX_MAX * m.brep_on1_P2[p, t, w]

    def _boostcap_on1_P2_lb(m, p, t, w):
        return m.w_boostcap_on1_P2[p, t, w] >= pipe_cap_P2(m, p, w) - QMAX_MAX * (1 - m.brep_on1_P2[p, t, w])

    m.BoostCapOn1_P2_ub1 = pyo.Constraint(m.P2_on, m.T, m.W, rule=_boostcap_on1_P2_ub1)
    m.BoostCapOn1_P2_ub2 = pyo.Constraint(m.P2_on, m.T, m.W, rule=_boostcap_on1_P2_ub2)
    m.BoostCapOn1_P2_lb  = pyo.Constraint(m.P2_on, m.T, m.W, rule=_boostcap_on1_P2_lb)

    def _boostcap_on2_P2_ub1(m, p, t, w):
        return m.w_boostcap_on2_P2[p, t, w] <= pipe_cap_P2(m, p, w)

    def _boostcap_on2_P2_ub2(m, p, t, w):
        return m.w_boostcap_on2_P2[p, t, w] <= QMAX_MAX * m.brep_on2_P2[p, t, w]

    def _boostcap_on2_P2_lb(m, p, t, w):
        return m.w_boostcap_on2_P2[p, t, w] >= pipe_cap_P2(m, p, w) - QMAX_MAX * (1 - m.brep_on2_P2[p, t, w])

    m.BoostCapOn2_P2_ub1 = pyo.Constraint(m.P2_on, m.T, m.W, rule=_boostcap_on2_P2_ub1)
    m.BoostCapOn2_P2_ub2 = pyo.Constraint(m.P2_on, m.T, m.W, rule=_boostcap_on2_P2_ub2)
    m.BoostCapOn2_P2_lb  = pyo.Constraint(m.P2_on, m.T, m.W, rule=_boostcap_on2_P2_lb)

    def _boostcap_off_P2_ub1(m, p, t, w):
        return m.w_boostcap_off_P2[p, t, w] <= pipe_cap_P2(m, p, w)

    def _boostcap_off_P2_ub2(m, p, t, w):
        return m.w_boostcap_off_P2[p, t, w] <= QMAX_MAX * m.brep_off_P2[p, t, w]

    def _boostcap_off_P2_lb(m, p, t, w):
        return m.w_boostcap_off_P2[p, t, w] >= pipe_cap_P2(m, p, w) - QMAX_MAX * (1 - m.brep_off_P2[p, t, w])

    m.BoostCapOff_P2_ub1 = pyo.Constraint(m.P2_off, m.T, m.W, rule=_boostcap_off_P2_ub1)
    m.BoostCapOff_P2_ub2 = pyo.Constraint(m.P2_off, m.T, m.W, rule=_boostcap_off_P2_ub2)
    m.BoostCapOff_P2_lb  = pyo.Constraint(m.P2_off, m.T, m.W, rule=_boostcap_off_P2_lb)

    # ------------------------------------------------------------------
    # 22) Booster CAPEX (incremental installation costs)
    # ------------------------------------------------------------------
    # Uses the same auxiliary variables as booster OPEX: w_boostcap_*.
    # Interpretation:
    #   w_boostcap_*(p, t, ...) equals the pipeline design capacity if a booster is installed by time t, and 0 otherwise.
    # The incremental CAPEX at time t is proportional to the increase in that capacity:
    #   delta = w_boostcap(t) - w_boostcap(t_prev)

    # P1 booster CAPEX as "increment" between time steps (scenario-independent)
    def _cap_boost_on1_P1_rule(m, p, t):
        if t == t0:
            return m.c_boost_on1_P1[p, t] == m.cins_boost[t] * m.w_boostcap_on1_P1[p, t]
        t_prev = _prev_year[t]
        return m.c_boost_on1_P1[p, t] == m.cins_boost[t] * (m.w_boostcap_on1_P1[p, t] - m.w_boostcap_on1_P1[p, t_prev])

    def _cap_boost_on2_P1_rule(m, p, t):
        if t == t0:
            return m.c_boost_on2_P1[p, t] == m.cins_boost[t] * m.w_boostcap_on2_P1[p, t]
        t_prev = _prev_year[t]
        return m.c_boost_on2_P1[p, t] == m.cins_boost[t] * (m.w_boostcap_on2_P1[p, t] - m.w_boostcap_on2_P1[p, t_prev])

    def _cap_boost_off_P1_rule(m, p, t):
        if t == t0:
            return m.c_boost_off_P1[p, t] == m.cins_boost[t] * m.w_boostcap_off_P1[p, t]
        t_prev = _prev_year[t]
        return m.c_boost_off_P1[p, t] == m.cins_boost[t] * (m.w_boostcap_off_P1[p, t] - m.w_boostcap_off_P1[p, t_prev])

    m.CapexBoostOn1_P1 = pyo.Constraint(m.P1_on,  m.T, rule=_cap_boost_on1_P1_rule)
    m.CapexBoostOn2_P1 = pyo.Constraint(m.P1_on,  m.T, rule=_cap_boost_on2_P1_rule)
    m.CapexBoostOff_P1 = pyo.Constraint(m.P1_off, m.T, rule=_cap_boost_off_P1_rule)

    # P2 booster CAPEX as "increment" between time steps (per scenario)
    def _cap_boost_on1_P2_rule(m, p, t, w):
        if t == t0:
            return m.c_boost_on1_P2[p, t, w] == m.cins_boost[t] * m.w_boostcap_on1_P2[p, t, w]
        t_prev = _prev_year[t]
        return m.c_boost_on1_P2[p, t, w] == m.cins_boost[t] * (m.w_boostcap_on1_P2[p, t, w] - m.w_boostcap_on1_P2[p, t_prev, w])

    def _cap_boost_on2_P2_rule(m, p, t, w):
        if t == t0:
            return m.c_boost_on2_P2[p, t, w] == m.cins_boost[t] * m.w_boostcap_on2_P2[p, t, w]
        t_prev = _prev_year[t]
        return m.c_boost_on2_P2[p, t, w] == m.cins_boost[t] * (m.w_boostcap_on2_P2[p, t, w] - m.w_boostcap_on2_P2[p, t_prev, w])

    def _cap_boost_off_P2_rule(m, p, t, w):
        if t == t0:
            return m.c_boost_off_P2[p, t, w] == m.cins_boost[t] * m.w_boostcap_off_P2[p, t, w]
        t_prev = _prev_year[t]
        return m.c_boost_off_P2[p, t, w] == m.cins_boost[t] * (m.w_boostcap_off_P2[p, t, w] - m.w_boostcap_off_P2[p, t_prev, w])

    m.CapexBoostOn1_P2 = pyo.Constraint(m.P2_on,  m.T, m.W, rule=_cap_boost_on1_P2_rule)
    m.CapexBoostOn2_P2 = pyo.Constraint(m.P2_on,  m.T, m.W, rule=_cap_boost_on2_P2_rule)
    m.CapexBoostOff_P2 = pyo.Constraint(m.P2_off, m.T, m.W, rule=_cap_boost_off_P2_rule)


    capex_boost = sum(m.c_boost_on1_P1[p, t] + m.c_boost_on2_P1[p, t] for p in m.P1_on for t in m.T) + sum(
        m.c_boost_off_P1[p, t] for p in m.P1_off for t in m.T
    ) + sum(
        m.prob[w]
        * (
            sum(m.c_boost_on1_P2[p, t, w] + m.c_boost_on2_P2[p, t, w] for p in m.P2_on for t in m.T)
            + sum(m.c_boost_off_P2[p, t, w] for p in m.P2_off for t in m.T)
        )
        for w in m.W
    )

    opex_boost = (
        # P1: scenario-independent
        sum(
            m.cop_boost[t] * (
                sum(m.w_boostcap_on1_P1[p, t] + m.w_boostcap_on2_P1[p, t] for p in m.P1_on)
                + sum(m.w_boostcap_off_P1[p, t] for p in m.P1_off)
            )
            for t in m.T
        )
        # P2: expected value across scenarios
        + sum(
            m.prob[w] * sum(
                m.cop_boost[t] * (
                    sum(m.w_boostcap_on1_P2[p, t, w] + m.w_boostcap_on2_P2[p, t, w] for p in m.P2_on)
                    + sum(m.w_boostcap_off_P2[p, t, w] for p in m.P2_off)
                )
                for t in m.T
            )
            for w in m.W
        )
    )

    el_boost = sum(
        m.prob[w]
        * (
            sum(m.cel_boost[t] * m.w_boost_on1[p, t, w] for p in m.P_on for t in m.T)
            + sum(m.cel_boost[t] * m.w_boost_on2[p, t, w] for p in m.P_on for t in m.T)
            + sum(m.cel_boost[t] * m.w_boost_off[p, t, w] for p in m.P_off for t in m.T)
        )
        for w in m.W
    )

    # Industrial revenue (expected)
    sale_revenue = sum(m.prob[w] * sum(m.qsale[k, t, w] * m.selling_revenue[t] for k in m.K for t in m.T) for w in m.W)

    # Objective: minimize expected total cost (revenue enters as negative cost)
    total_cost = (
        capex_pipe_on
        + capex_pipe_off
        + opex_pipe_on
        + opex_pipe_off
        + ship_cost
        + capture_cost
        + injection_cost
        # + penalty_uncaptured
        + capex_init
        + opex_init
        + el_init
        + capex_boost
        + opex_boost
        + el_boost
        - sale_revenue
    )

    m.SystemCost = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    return m
