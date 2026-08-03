# developed_main.py

# --------------------------------------------------------------
# 1.  Standard imports
# --------------------------------------------------------------
import sys
import os, pathlib
from pyomo import environ as pyo
from pyomo.environ import value
from pyomo.opt import TerminationCondition as tc

# Local modules (relative imports inside package ‘model’)
from iberian_co2_network.data  import DATA
from iberian_co2_network.developed_plots import plot_network, save_all_plots
from iberian_co2_network import developed_solution, utils
from iberian_co2_network.developed_model import build_model

print("🔍  Current working directory:", os.getcwd())
print("🔍  Script location:", pathlib.Path(__file__).resolve())

# --------------------------------------------------------------
# 2.  Build model
# --------------------------------------------------------------
print("Building model …")
m = build_model(DATA)  # dict, not module

# Diagnostic
print(f"Variables   : {m.nvariables()}")
print(f"  ‣ binaries : {sum(1 for v in m.component_data_objects(pyo.Var) if v.is_binary())}")
print(f"Constraints : {m.nconstraints()}")

# --------------------------------------------------------------
# 3.  Configure Gurobi solver (two-phase solve)
# --------------------------------------------------------------
print("Launching Gurobi …")
solver = pyo.SolverFactory("gurobi")

# Export LP for IIS analysis (optional but useful)
m.write("full_model.lp", io_options={"symbolic_solver_labels": True})
print("✔  Model exported to", os.path.abspath("full_model.lp"))

# ---- Shared/base options
BASE_OPTS = {
    "Seed": 1,
    "SoftMemLimit": 6144,     # 6 GB -> 6144 MB
    "NodeFileStart": 0.5,     # start paging nodes to disk after 0.5 GB
    "Cuts": 2,
    "Presolve": 2,
    "Symmetry": 2,
}

# ---- Phase 1 (fast incumbent)
PHASE1_OPTS = {
    **BASE_OPTS,
    "TimeLimit": 30_000,      # adjust if you want
    "MIPGap": 0.1,           # <-- phase 1 gap
    "MIPFocus": 1,
    "Heuristics": 0.3,
}

# ---- Phase 2 (refine)
PHASE2_OPTS = {
    **BASE_OPTS,
    "TimeLimit": 150_000,     # original time limit
    "MIPGap": 0.03,           # <-- phase 2 gap
    "MIPFocus": 1,
    "Heuristics": 0.3,
}

# --------------------------------------------------------------
# 3a) Phase 1 solve
# --------------------------------------------------------------
print("\n================ PHASE 1 ================")
solver.options.clear()
solver.options.update(PHASE1_OPTS)

results1 = solver.solve(m, tee=True)
term1 = results1.solver.termination_condition
print("Phase 1 termination:", term1)

# If Phase 1 is infeasible/unbounded, stop (warm-start makes no sense)
if term1 in (tc.infeasible, tc.unbounded, tc.infeasibleOrUnbounded):
    print("⚠️  Phase 1 reported infeasible/unbounded. Running infeasibility logger...")
    from pyomo.util.infeasible import log_infeasible_constraints
    import logging
    logging.getLogger('pyomo').setLevel(logging.INFO)
    log_infeasible_constraints(m, tol=1e-6, log_expression=True)
    sys.exit(1)

# --------------------------------------------------------------
# 3b) Phase 2 solve (warm-start from Phase 1 incumbent)
# --------------------------------------------------------------
print("\n================ PHASE 2 ================")
solver.options.clear()
solver.options.update(PHASE2_OPTS)

# warmstart=True tells Pyomo to pass current variable values as an initial solution (when supported)
results2 = solver.solve(m, tee=True, warmstart=True)
term2 = results2.solver.termination_condition
print("Phase 2 termination:", term2)

# --------------------------------------------------------------
# 4.  Check termination and report
# --------------------------------------------------------------
# With a MIPGap, you may get 'optimal' or just 'feasible' depending on the interface;
# accept feasible solutions, but reject infeasible/unbounded.
if term2 in (tc.infeasible, tc.unbounded, tc.infeasibleOrUnbounded):
    print("⚠️  Phase 2 reported infeasible/unbounded. Running infeasibility logger...")
    from pyomo.util.infeasible import log_infeasible_constraints
    import logging
    logging.getLogger('pyomo').setLevel(logging.INFO)
    log_infeasible_constraints(m, tol=1e-6, log_expression=True)
    sys.exit(1)

if term2 != tc.optimal and term2 != tc.feasible:
    print("⚠️  Solver did not reach optimality/feasibility as expected. Condition:", term2)
    # You can still continue if you want; here we stop to be safe:
    sys.exit(1)

total_cost = pyo.value(m.SystemCost)
print(f"\n✅ Solution accepted. Expected cost = {total_cost:,.0f} M€")

# --------------------------------------------------------------
# 5. Post-processing
# --------------------------------------------------------------
W = list(m.W.data())  # scenarios
T = list(m.T.data())  # years

# 1) Capture, storage, and sales to industry (by t and w)
capture = {
    t: {w: sum(value(m.qcap[e, t, w]) for e in m.E) for w in W}
    for t in T
}
storage = {
    t: {w: sum(value(m.qstore[s, t, w]) for s in m.S) for w in W}
    for t in T
}
utilization = {
    t: {w: sum(value(m.qsale[k, t, w]) for k in m.K) for w in W}
    for t in T
}

# 2) Flows per pipeline (on/off), by (p, t, w)
q_on = {(p, t, w): value(m.q_on[p, t, w]) for p in m.P_on for t in T for w in W}
q_off = {(p, t, w): value(m.q_off[p, t, w]) for p in m.P_off for t in T for w in W}

# 3) Shipping (offshore), by (p, t, w)
b_ship = {(p, t, w): value(m.b_ship[p, t, w]) for p in m.P_off for t in T for w in W}
nship = {(p, t, w): value(m.nship[p, t, w]) for p in m.P_off for t in T for w in W}

# 4) Pressures (by node/pipe), by (·, t, w)
pi_node = {(i, t, w): value(m.pi_node[i, t, w]) for i in m.N for t in T for w in W}
pi_orig = {(p, t, w): value(m.pi_orig[p, t, w]) for p in m.P for t in T for w in W}
pi_dest = {(p, t, w): value(m.pi_dest[p, t, w]) for p in m.P for t in T for w in W}

# 5) Dominance and dominant flow (by (·, t, w))
y_on = {(p, t, w): value(m.y_on[p, t, w]) for p in m.P_on for t in T for w in W}
y_off = {(p, t, w): value(m.y_off[p, t, w]) for p in m.P_off for t in T for w in W}
q_dom = {(i, t, w): value(m.q_dom[i, t, w]) for i in m.N for t in T for w in W}

# --------------------------------------------------------------
# P1 variables: NO scenario index 'w'
# --------------------------------------------------------------

# 6) Construction decisions
#    - P1 WITHOUT w
#    - P2 WITH w
z_on_P1 = {(p, t): value(m.z_on_P1[p, t]) for p in m.P1_on for t in T}
z_off_P1 = {(p, t): value(m.z_off_P1[p, t]) for p in m.P1_off for t in T}

z_on_P2 = {(p, t, w): value(m.z_on_P2[p, t, w]) for p in m.P2_on for t in T for w in W}
z_off_P2 = {(p, t, w): value(m.z_off_P2[p, t, w]) for p in m.P2_off for t in T for w in W}

# 7) Selected diameters
#    - P1 WITHOUT w
#    - P2 WITH w
b_diam_on_P1 = {(p, d): value(m.b_diam_on_P1[p, d]) for p in m.P1_on for d in m.D}
b_diam_off_P1 = {(p, d): value(m.b_diam_off_P1[p, d]) for p in m.P1_off for d in m.D}

b_diam_on_P2 = {(p, d, w): value(m.b_diam_on_P2[p, d, w]) for p in m.P2_on for d in m.D for w in W}
b_diam_off_P2 = {(p, d, w): value(m.b_diam_off_P2[p, d, w]) for p in m.P2_off for d in m.D for w in W}

# 8) Boosters
#    - P1 WITHOUT w
#    - P2 WITH w
brep_on1_P1 = {(p, t): value(m.brep_on1_P1[p, t]) for p in m.P1_on for t in T}
brep_on2_P1 = {(p, t): value(m.brep_on2_P1[p, t]) for p in m.P1_on for t in T}
brep_off_P1 = {(p, t): value(m.brep_off_P1[p, t]) for p in m.P1_off for t in T}

brep_on1_P2 = {(p, t, w): value(m.brep_on1_P2[p, t, w]) for p in m.P2_on for t in T for w in W}
brep_on2_P2 = {(p, t, w): value(m.brep_on2_P2[p, t, w]) for p in m.P2_on for t in T for w in W}
brep_off_P2 = {(p, t, w): value(m.brep_off_P2[p, t, w]) for p in m.P2_off for t in T for w in W}

# 9) Initial compression decisions:
z_init = {(e, t, w): value(m.z_init[e, t, w]) for e in m.E for t in T for w in W}

# --------------------------------------------------------------
# Export results
# --------------------------------------------------------------
developed_solution.export_results(m)

# 1) Save the 3x5 = 15 plots (without showing them on screen)
save_all_plots(m, DATA)  # uses m.THETA if available to name the folder

# 2) (Optional) Show a single “reference” plot on screen
#    e.g., base scenario in 2050, and also save it
plot_network(
    m, DATA,
    scenario="base_utilization",
    year=2050,
    save=True,
    show=True
)

print("\n🎉 Run completed. Results in /output_developed_model")
