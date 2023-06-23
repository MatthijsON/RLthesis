from typing import NamedTuple
from distillation import ROOT_DIR, os

import chex
import jax.numpy as jnp
import jax
from jax.scipy import linalg
from jax import lax, grad
# import timeit

components=['n-Butane', 'n-Pentane', 'n-Octane']
flow_rate_tol = jnp.array(1.0e-4)
temperature_tol = jnp.array(1.0e-2)
dampening_factor = jnp.array(1.0)
N_max = jnp.array(100)
z_feed1 = jnp.array([0.2, 0.5, 0.3])
num_constants = jnp.array(5)

# def add_parameters(verbose=False):
#     from distillation.equilibrium_data.depriester_charts import DePriester
#     from distillation.equilibrium_data.heat_capacity_liquid import CpL
#     from distillation.equilibrium_data.heat_capacity_vapor import CpV
#     from distillation.equilibrium_data.heats_of_vaporization import dH_vap
#
#     K_func = {
#             key: DePriester(key, verbose) for key in components
#         }
#     CpL_func = {
#             key: CpL(key, verbose) for key in components
#         }
#     CpV_func = {
#             key: CpV(key, verbose) for key in components
#         }
#     dH_func = {
#             key: dH_vap(key, verbose) for key in components
#         }
#     T_ref = {
#             key: val.T_ref for key, val in dH_func.items()
#         }
#     return K_func, CpL_func, CpV_func, dH_func, T_ref
#
# start = timeit.default_timer()
#
# K_func = add_parameters(verbose=False)[0]
# CpL_func = add_parameters(verbose=False)[1]
# CpV_func = add_parameters(verbose=False)[2]
# dH_func = add_parameters(verbose=False)[3]
# T_ref = add_parameters(verbose=False)[4]
#
#
#
# time_taken = timeit.default_timer() - start
# print('TIME TAKEN IS EQUAL TO ---------------', time_taken)

K_func = {}
CpL_func = {}
CpV_func = {}
dH_func = {}
T_ref = {}

component_indices = {
    'n-Butane': 2,
    'n-Pentane': 3,
    'n-Octane': 4
}

path_depriester = os.path.join(ROOT_DIR, 'equilibrium_data', 'depriester.csv')
path_CpL = os.path.join(ROOT_DIR, 'equilibrium_data', 'heat_capacity_liquid.csv')
path_dH = os.path.join(ROOT_DIR, 'equilibrium_data', 'heats_of_vaporization.csv')

with open(path_depriester, 'r') as f:
    next(f) # skip header row
    for line in f:
        fields = line.strip().split(',')
        name = fields[0]
        if name in component_indices:
            values = jnp.array(list(map(float, fields[1:-1])))
            K_func[name] = values


with open(path_CpL, 'r') as f:
    next(f) # skip header row
    for line in f:
        fields = line.strip().split(',')
        name = fields[0]
        values = jnp.array(list(map(float, fields[2:-3])))
        CpL_func[name] = values

CpV_value = jnp.array([4.*8.314*1000])
CpV_func = {key: CpV_value for key in components}

with open(path_dH, 'r') as f:
    next(f) # skip header row
    for line in f:
        fields = line.strip().split(',')
        name = fields[0]
        dH_values = jnp.array(list(map(float, fields[2:3])))
        T_ref_values = jnp.array(list(map(float, fields[3:4])))
        dH_func[name] = dH_values
        T_ref[name] = T_ref_values

class State(NamedTuple):
    F_feed: chex.Array
    P_feed: chex.Array
    z_feed: dict
    RR: chex.Array
    Distillate: chex.Array
    N: chex.Array
    feed_stage: chex.Array
    T_feed_guess: chex.Array
    T_feed: chex.Array
    L: chex.Array
    V: chex.Array
    L_old: chex.Array
    V_old: chex.Array
    F: chex.Array
    z: dict
    l: dict
    T: chex.Array
    T_old: chex.Array
    K: dict
    bubble_point: chex.Array
    eps: chex.Array
    A: chex.Array
    B: chex.Array
    C: chex.Array
    D: chex.Array
    E: chex.Array
    BE: chex.Array
    CE: chex.Array
    DE: chex.Array
    EE: chex.Array

state = State(
    F_feed= jnp.array(1000.0),
    P_feed= jnp.array(2.0*1e5),
    z_feed= {key: val for key, val in zip(components, z_feed1)},
    RR= jnp.array(1.0),
    Distillate= jnp.array(400.0),
    N= jnp.array(30),
    feed_stage= jnp.array(15),
    T_feed_guess= jnp.array(300.0),
    T_feed= jnp.zeros(1),
    L= jnp.zeros(N_max+1),
    V= jnp.zeros(N_max+1),
    L_old= jnp.zeros(N_max+1),
    V_old= jnp.zeros(N_max+1),
    F= jnp.zeros(N_max+1),
    z= {key: jnp.zeros(N_max+1) for key in components},
    l= {key: jnp.zeros(N_max+1) for key in components},
    T= jnp.zeros(N_max+1),
    T_old= jnp.zeros(N_max+1),
    K= {key: jnp.zeros(N_max+1) for key in components},
    bubble_point= jnp.zeros(N_max+1),
    eps= jnp.zeros(N_max+1),
    A= -1 * jnp.ones(N_max),
    B= jnp.zeros(N_max+1),
    C= jnp.zeros(N_max),
    D= jnp.zeros(N_max+1),
    E= jnp.zeros((N_max +1)**2).reshape(N_max+1,N_max+1),
    BE= jnp.zeros(N_max+1),
    CE= jnp.zeros(N_max+1),
    DE= jnp.zeros(N_max+1),
    EE= jnp.zeros((N_max)**2).reshape(N_max,N_max)
)

#state = state._replace(z = z_new)
'replacements'
# for component in components:
#     state.z[component] = state.z[component].at[state.feed_stage].set(state.z_feed[component])

new_z = {component:state.z[component].at[state.feed_stage].set(state.z_feed[component]) for component in components}
state = state._replace(z = new_z)
# state.z[component].at[state.feed_stage].set(state.z_feed[component]) for component in components)
#state = state._replace(z = state.z[component].at[state.feed_stage].set(state.z_feed[component]) for component in components)
state = state._replace(F = state.F.at[state.feed_stage].set(state.F_feed))

'Constant evaluation functions'
'functions within imports'
def Rankine_to_Kelvin(T):
    return T / 1.8

def Kelvin_to_Rankine(T):
    return T * 1.8

def psia_to_Paa(P):
    return P / 14.5038 * 1e5

def Paa_to_psia(P):
    return P / 1e5 * 14.5038

def eval_SI_Depriester(T, p, c):
    return eval_Depriester(
        Kelvin_to_Rankine(T), Paa_to_psia(p), c)

def eval_Depriester(T, p, c):
    return jnp.exp(
        K_func[c][0] / T / T + K_func[c][1] / T + K_func[c][2] + K_func[c][3] * jnp.log(p) + K_func[c][4] / p / p + K_func[c][5] / p
    )

def eval_CpL(T, c):
    constants = CpL_func[c]
    power_law = [constants[i]*jax.lax.pow(T, (i-1.0)) for i in range(1, num_constants + 1)]
    power_law_array = jnp.array(power_law)
    result = jnp.sum(power_law_array)
    return result

def integral_CpL(T, c):
    constants = jnp.array(CpL_func[c])
    power_law = [constants[i-1] * jax.lax.pow(T, (i - 0.0))/i for i in range(1, num_constants + 1)]
    power_law_array = jnp.array(power_law)
    result = jnp.sum(power_law_array)
    return result

def integral_dT_CpL(T_ref, T, c):
    int_T = integral_CpL(T, c)
    int_T_ref = integral_CpL(T_ref, c)
    result = int_T - int_T_ref
    return result

def integral_dT_CpV(T_ref, T, c):
    value = CpV_func[c]
    result = value * (T - T_ref)
    return result

def eval_dH_func(c):
    value = dH_func[c]
    result = value * 1e6
    return result

#
# def make_ABC_condenser(state: State, component) -> State:
#     assert abs(state.V[0]) < 1e-8
#     'total condenser'
#     new_state = state._replace(B = state.B.at[0].set(1 + state.Distillate / state.L[0]),
#                                C = state.C.at[0].set(-state.V[1] * state.K[component][1] / state.L[1]),
#                                D = state.D.at[0].set(state.F[0] * state.z[component][0]))
#     return new_state
#
# def make_ABC_reboiler(state: State, component) -> State:
#     'reboiler'
#     Bottoms = state.F_feed - state.Distillate
#     B_new = 1.0 + state.V[state.N] * state.K[component][state.N] / Bottoms
#     D_new = state.F[state.N] * state.z[component][state.N]
#     new_state = state._replace(B = state.B.at[state.N].set(B_new),
#                                D = state.D.at[state.N].set(D_new))
#     return new_state
#
# def make_ABC_stages(state: State, component) -> State:
#     'stages'
#     B_new = 1 + state.V[1:state.N] * state.K[component][1:state.N] / state.L[1:state.N]
#     C_new = -state.V[2:(state.N+1)] * state.K[component][2:(state.N+1)] / state.L[2:(state.N+1)]
#     D_new = state.F[1:state.N] * state.z[component][1:state.N]
#     new_state = state._replace(B = state.B.at[1:state.N].set(B_new),
#                                C = state.C.at[1:state.N].set(C_new),
#                                D = state.D.at[1:state.N].set(D_new))
#     return new_state
#
# def make_ABC(state: State, component):
#     return make_ABC_stages(state, component), make_ABC_reboiler(state, component), make_ABC_condenser(state, component)

def make_ABC(state: State, component):
    assert abs(state.V[0]) < 1e-8
    B_condenser = 1 + state.Distillate / state.L[0]
    C_condenser = -state.V[1] * state.K[component][1] / state.L[1]
    D_condenser = state.F[0] * state.z[component][0]

    Bottoms = state.F_feed - state.Distillate
    B_reboiler = 1.0 + state.V[state.N] * state.K[component][state.N] / Bottoms
    D_reboiler = state.F[state.N] * state.z[component][state.N]

    B_stages = 1 + state.V[1:state.N] * state.K[component][1:state.N] / state.L[1:state.N]
    C_stages = -state.V[2:(state.N+1)] * state.K[component][2:(state.N+1)] / state.L[2:(state.N+1)]
    D_stages = state.F[1:state.N] * state.z[component][1:state.N]

    new_state = state._replace(B = state.B.at[0].set(B_condenser).at[state.N].set(B_reboiler).at[1:state.N].set(B_stages),
                               C = state.C.at[0].set(C_condenser).at[1:state.N].set(C_stages),
                               D = state.D.at[0].set(D_condenser).at[state.N].set(D_reboiler).at[1:state.N].set(D_stages))
    return new_state
# for i in components:
#     solve_component_mass_bal(i)

# make_ABC(state,components[0])
# print(state.B)

def h_pure_rule(c, T):
    integral = integral_dT_CpL(T_ref[c], T, c)
    return integral

def h_j_rule(state: State, stage):
    rule_result = [x_ij_expr(state, c , stage) * h_pure_rule(c, state.T[stage]) for c in components]
    rule_result_array = jnp.array(rule_result)
    result = jnp.sum(rule_result_array)
    return result

def x_ij_expr(state: State, i, j):
    return state.l[i][j] / state.L[j]

def h_feed_rule(state: State, stage):
    rule_result = [state.z[c][stage] * h_pure_rule(c, state.T_feed_guess) for c in components]
    rule_result_array = jnp.array(rule_result)
    result = jnp.sum(rule_result_array)
    return result

def H_pure_rule(c, T):
    integral = integral_dT_CpV(T_ref[c], T, c)
    dh = eval_dH_func(c)
    return integral + dh

def H_j_rule(state: State, stage):
    rule_result = [y_ij_expr(state, c, stage) * H_pure_rule(c, state.T[stage]) for c in components]
    rule_result_array = jnp.array(rule_result)
    result = jnp.sum(rule_result_array)
   # return jnp.sum(y_ij_expr(state, c, stage) * H_pure_rule(c, state.T[stage]) for c in components)
    return result
def y_ij_expr(state: State, i, j):
    # p_input = state.P_feed * x_ij_expr(state, i, j)
    evalSI = eval_SI_Depriester(state.T[j], state.P_feed, i)
    result = evalSI * x_ij_expr(state, i, j)
    return result

def Q_condenser_rule(state: State):
    return state.Distillate * (1 + state.RR) * (h_j_rule(state, 0) - H_j_rule(state, 1))

def Q_reboiler_rule(state: State):
    return state.Distillate * h_j_rule(state, 0) + (state.F_feed - state.Distillate) * h_j_rule(state, state.N) \
            - state.F_feed * h_feed_rule(state, state.feed_stage) - Q_condenser_rule(state)

def step_3_to_step_6(state: State):
    num_iter = 0
    while not T_is_converged(state):
        update_K_values(state)
        for i in components:
            solve_component_mass_bal(state, i)
        update_T_values(state)
        num_iter += 1
    print('while loop exits with %i iterations' % num_iter)

def run(state: State):
    #generate_initial_guess(state)
    step_3_to_step_6(state)
    solve_energy_balances_vapor(state)
    solve_energy_balances_liquid(state)
    main_loop = 0
    while not flow_rates_converged(state):
        for i in components:
            solve_component_mass_bal(state, i)
        update_T_values(state)
        step_3_to_step_6(state)
        solve_energy_balances_vapor(state)
        solve_energy_balances_liquid(state)
        main_loop += 1
        print(main_loop)

def update_K(state: State, component):
    new_K_component = state.K[component].copy()
    new_K_component = new_K_component.at[:state.N + 1].set(eval_SI_Depriester(state.T[:state.N + 1], state.P_feed, component))
    return new_K_component

def update_K_values(state: State):
    # new_K = {component: state.K[component].at[:state.N+1].set(eval_SI_Depriester(state.T[:state.N+1],state.P_feed,component)) for component in components}
    new_K = {component: update_K(state, component) for component in components}
    new_state = state._replace(K= new_K,
                               T_old= state.T_old.at[:state.N+1].set(state.T[:state.N+1])
                               )
    return new_state

def update_T_values(state: State):
    new_T_values = [state.T_old[i] + dampening_factor * (bubble_T(state, i) - state.T_old[i]) for i in range(state.N + 1)]
    new_T_array = jnp.array(new_T_values)
    new_T = state.T.at[:state.N+1].set(new_T_array)
    # new_T_replace = (state.T.at[i].set(state.T_old[i] + dampening_factor * (bubble_T(state, i) - state.T_old[i])) for i in range(state.N + 1))
    new_state = state._replace(T= new_T)

    return new_state

def bubble_T(state: State, stage):
    # l_total = jnp.sum(state.l[c][stage] for c in components)
    l_component = [state.l[c][stage] for c in components]
    l_component_array = jnp.array(l_component)
    l_total = jnp.sum(l_component_array)
    # K_vals = [K_func[c].eval_SI for c in components]
    # K_vals = [eval_SI_Depriester(state.T,state.P_feed,state.Distillate) ] #change input variables, currently done to avoid warnings
    x_vals = jnp.array([state.l[c][stage] / l_total for c in components])
    T_bubble = bubble_point(x_vals, state.P_feed, state.T_old[stage])
    return T_bubble

def calculate_T_feed(state: State):
    new_T_feed = bubble_T_feed(state)
    new_T = initialize_stage_temperatures(state)
    new_state = state._replace(T_feed= new_T_feed,
                               T= new_T
                               )
    return new_state

def initialize_stage_temperatures(state: State):
    result = state.T.at[0:state.N].set(state.T_feed)
    return result

def bubble_T_feed(state: State):
    x_vals = jnp.array([state.z_feed[c] for c in components])
    T_bubble = bubble_point(x_vals, state.P_feed, state.T_feed_guess)
    return T_bubble

def initialize_flow_rates(state: State):
    # new_L_above_feed = state._replace(L= state.L.at[:state.feed_stage].set(state.RR * state.Distillate))
    # new_L_under_feed = state._replace(L=state.L.at[state.feed_stage:state.N].set(state.RR * state.Distillate + state.F_feed))
    # new_L_at_N = state._replace(L=state.L.at[state.N].set(state.F_feed - state.Distillate))
    # new_V = state._replace(V=state.V.at[1:state.N].set(state.RR * state.Distillate + state.Distillate))
    new_L_above_feed = state.RR * state.Distillate
    new_L_under_feed = state.RR * state.Distillate + state.F_feed
    new_L_at_N = state.F_feed - state.Distillate
    new_V = state.RR * state.Distillate + state.Distillate
    new_state = state._replace(V= state.V.at[1:state.N].set(new_V),
                               L= state.L.at[:state.feed_stage].set(new_L_above_feed).at[state.feed_stage:state.N].set(new_L_under_feed).at[state.N].set(new_L_at_N))

    return new_state

def T_is_converged(state: State):
    eps = jnp.abs(state.T - state.T_old)
    return eps.max() < temperature_tol

def solve_component_mass_bal(state: State, component):
    state = make_ABC(state, component)
    new_l = state.l.copy()
    new_l[component] = new_l[component].at[:state.N+1].set(solve_diagonal(state))
    # new_l = {component: state.l[component].at[:state.N+1].set(solve_diagonal(state))}
    # new_z = {component: state.z[component].at[state.feed_stage].set(state.z_feed[component]) for component in
    #          components}
    new_state = state._replace(l= new_l)
    return new_state

def update_flow_rates(state: State):
    # l_values = [(state.l[c][i] for c in components) for i in range(state.N + 1)]
    # l_values_array = jnp.array(l_values)
    # L_values = jnp.sum(l_values_array)
    # L_per_stage = (state.L.at[i].set(jnp.sum(state.l[c][i] for c in components)) for i in range(state.N + 1))
    # new_L = state._replace(L= L_values)
    # new_V_top = state._replace(V= state.V.at[0].set(0))
    # V_other = state.V.at[2:state.N+1].set(V_values)
    # V_other = (state.V.at[i].set(state.L[i-1] + state.Distillate - F_sum) for i in range(2, (state.N+1)))

    l_values = jnp.stack([state.l[c] for c in components])
    L_values = jnp.sum(l_values, axis=0)
    new_V_condenser = (state.RR + 1) * state.Distillate
    F_values = [state.F[i] for i in range(2, (state.N+1))]
    F_values_array = jnp.array(F_values)
    F_sum = jnp.sum(F_values_array)
    V_values = [state.L[i-1] + state.Distillate - F_sum for i in range(2, (state.N+1))]
    V_other = jnp.array(V_values)
    new_state = state._replace(L= L_values,
                               V= state.V.at[0].set(0).at[1].set(new_V_condenser).at[2:state.N+1].set(V_other)
                               )
    return new_state

def solve_energy_balances_stream_copy(state: State):
    new_state = state._replace(L_old= state.L_old.at[:state.N].set(state.L[:state.N]),
                               V_old= state.V_old.at[:state.N].set(state.V[:state.N])
                               )
    return new_state

def solve_energy_balances_condenser(state: State):
    CE_new = h_j_rule(state, 0) - H_j_rule(state, 1)
    DE_new = state.F[0] * h_feed_rule(state, 0) + Q_condenser_rule(state)
    new_state = state._replace(BE= state.BE.at[0].set(0),
                               CE= state.CE.at[0].set(CE_new),
                               DE= state.DE.at[0].set(DE_new)
                               )
    return new_state

#
# def solve_energy_balances_stages(state: State):
#     BE_new_values = [H_j_rule(state, j) - h_j_rule(state, (j - 1)) for j in range(1, state.N)]
#     BE_new_values_array = jnp.array(BE_new_values)
#     BE_new = state.BE.at[1:state.N].set(BE_new_values_array)
#     start = timeit.default_timer()
#     BE_new_values_test = jnp.array([H_j_rule(state, j) - h_j_rule(state, (j - 1)) for j in range(0, N_max + 1)])
#     stop = timeit.default_timer() - start
#     print(stop)
#     start2 = timeit.default_timer()
#     BE_new_values_test_vmap = jax.vmap(lambda j: H_j_rule(state, j))(jnp.arange(N_max + 1)) - jax.vmap(
#         lambda j: h_j_rule(state, j - 1))(jnp.arange(N_max + 1))
#     stop2 = timeit.default_timer() - start2
#     print(stop2)
#
#     def BE_function(i):
#         result = H_j_rule(state, i) - h_j_rule(state, (i - 1))
#         return result
#
#     BE_new_values_test_vmap2 = jax.vmap(BE_function)(jnp.arange(N_max + 1))
#
#     mask = jnp.logical_and(jnp.arange(N_max + 1) > 0, jnp.arange(N_max + 1) < state.N)
#     BE_new_test = jnp.where(mask, BE_new_values_test, 0)
#
#     CE_new_values_test = jnp.array([h_j_rule(state, j) - H_j_rule(state, (j + 1)) for j in range(0, N_max + 1)])
#
#     def CE_function(i):
#         result = h_j_rule(state, i) - H_j_rule(state, (i + 1))
#         return result
#
#     CE_new_values_vmap = jax.vmap(CE_function)(jnp.arange(N_max + 1))
#
#     CE_new_test = jnp.where(mask, CE_new_values_test, 0)
#
#     CE_new_values = [h_j_rule(state, j) - H_j_rule(state, (j + 1)) for j in range(1, state.N)]
#     CE_new_values_array = jnp.array(CE_new_values)
#     CE_new = state.CE.at[1:state.N].set(CE_new_values_array)
#
#     DE_sum_1_F_values_test = jnp.array([state.F[i] for i in range(0, N_max + 1)])
#     DE_sum_1_rule_values_test = jnp.array([h_j_rule(state, i) for i in range(0, N_max + 1)])
#     mask2 = jnp.logical_and(jnp.arange(N_max + 1) > 1, jnp.arange(N_max + 1) < state.N + 1)
#     DE_sum_1_F_test = jnp.where(mask2, DE_sum_1_F_values_test, 0)
#     DE_sum_1_rule_test = jnp.where(mask, DE_sum_1_rule_values_test, 0)
#     DE_sum_1_values_test = DE_sum_1_F_test[2:] * DE_sum_1_rule_test[1:-1]
#     DE_sum_1_test = jnp.sum(DE_sum_1_values_test)
#
#     DE_sum_1_F_values = [state.F[i] for i in range(2, state.N + 1)]
#     DE_sum_1_F_values_array = jnp.array(DE_sum_1_F_values)
#     DE_sum_1_rule_values = [h_j_rule(state, i) for i in range(1, state.N)]
#     DE_sum_1_rule_values_array = jnp.array(DE_sum_1_rule_values)
#     DE_sum_1_values = DE_sum_1_F_values_array * DE_sum_1_rule_values_array
#     DE_sum_1 = jnp.sum(DE_sum_1_values)
#
#     DE_sum_2_F_values_test = jnp.array([state.F[i] for i in range(0, N_max + 1)])
#     DE_sum_2_rule_values_test = jnp.array([h_j_rule(state, i) for i in range(0, N_max + 1)])
#     mask3 = jnp.arange(N_max + 1) < (state.N - 1)
#     DE_sum_2_F_test = jnp.where(mask, DE_sum_2_F_values_test, 0)
#     DE_sum_2_rule_test = jnp.where(mask3, DE_sum_2_rule_values_test, 0)
#     DE_sum_2_values_test = DE_sum_2_F_test[1:] * DE_sum_2_rule_test[:-1]
#     DE_sum_2_test = jnp.sum(DE_sum_2_values_test)
#
#     DE_sum_2_F_values = [state.F[i] for i in range(1, state.N)]
#     DE_sum_2_F_values_array = jnp.array(DE_sum_2_F_values)
#     DE_sum_2_rule_values = [h_j_rule(state, i) for i in range((state.N - 1))]
#     DE_sum_2_rule_values_array = jnp.array(DE_sum_2_rule_values)
#     DE_sum_2_values = DE_sum_2_F_values_array * DE_sum_2_rule_values_array
#     DE_sum_2 = jnp.sum(DE_sum_2_values)
#
#     DE_new_values_test = jnp.array([state.F[i] * h_feed_rule(state, i) - state.Distillate * (
#                 h_j_rule(state, (i - 1)) - h_j_rule(state, i)) - DE_sum_1_test + DE_sum_2_test for i in
#                                     range(0, N_max + 1)])
#     DE_new_values_test2 = jnp.where(mask, DE_new_values_test, 0)
#
#     DE_new_values = [state.F[i] * h_feed_rule(state, i) - state.Distillate * (
#                 h_j_rule(state, (i - 1)) - h_j_rule(state, i)) - DE_sum_1 + DE_sum_2 for i in range(1, state.N)]
#     DE_new = state.DE.at[1:state.N].set(DE_new_values)
#     new_state = state._replace(BE=BE_new,
#                                CE=CE_new,
#                                DE=DE_new)
#     return new_state
def solve_energy_balances_stages(state: State):
    BE_new_values = [H_j_rule(state, j) - h_j_rule(state, (j-1)) for j in range(1, state.N)]
    BE_new_values_array = jnp.array(BE_new_values)
    BE_new = state.BE.at[1:state.N].set(BE_new_values_array)
    # BE_new = (state.BE.at[j].set(H_j_rule(state, j) - h_j_rule(state, (j-1))) for j in range(1, state.N))

    CE_new_values = [h_j_rule(state, j) - H_j_rule(state, (j+1)) for j in range(1, state.N)]
    CE_new_values_array = jnp.array(CE_new_values)
    CE_new = state.CE.at[1:state.N].set(CE_new_values_array)
    # CE_new = (state.CE.at[j].set(h_j_rule(state, j) - H_j_rule(state, (j+1))) for j in range(1, state.N))

    DE_sum_1_F_values = [state.F[i] for i in range(2, state.N+1)]
    DE_sum_1_F_values_array = jnp.array(DE_sum_1_F_values)
    DE_sum_1_rule_values = [h_j_rule(state, i) for i in range(1, state.N)]
    DE_sum_1_rule_values_array = jnp.array(DE_sum_1_rule_values)
    DE_sum_1_values = DE_sum_1_F_values_array * DE_sum_1_rule_values_array
    DE_sum_1 = jnp.sum(DE_sum_1_values)

    DE_sum_2_F_values = [state.F[i] for i in range(1, state.N)]
    DE_sum_2_F_values_array = jnp.array(DE_sum_2_F_values)
    DE_sum_2_rule_values = [h_j_rule(state, i) for i in  range((state.N-1))]
    DE_sum_2_rule_values_array = jnp.array(DE_sum_2_rule_values)
    DE_sum_2_values = DE_sum_2_F_values_array * DE_sum_2_rule_values_array
    DE_sum_2 = jnp.sum(DE_sum_2_values)

    DE_new_values = [state.F[i] * h_feed_rule(state, i) - state.Distillate * (h_j_rule(state, (i-1)) - h_j_rule(state, i)) - DE_sum_1 + DE_sum_2 for i in range(1, state.N)]
    DE_new = state.DE.at[1:state.N].set(DE_new_values)
    # DE_new = (state.DE.at[j].set(state.F[j] * h_feed_rule(state,j) - state.Distillate * (h_j_rule(state,(j-1)) - h_j_rule(state,j)) \
    #                              - sum(state.F[k] for k in range(j + 1) * h_j_rule(state, j)) \
    #                              + sum(state.F[k] for k in range(j)) * h_j_rule(state, (j-1))) for j in range(1, state.N))
    new_state = state._replace(BE= BE_new,
                               CE= CE_new,
                               DE= DE_new)
    return new_state

def solve_energy_balances_reboiler(state: State):
    BE_new = state.BE.at[state.N].set(H_j_rule(state, state.N) - h_j_rule(state, (state.N - 1)))
    DE_new = state.DE.at[state.N].set(state.F[state.N] * h_feed_rule(state, state.N) + Q_reboiler_rule(state) \
                                - (state.F_feed - state.Distillate) * (h_j_rule(state, state.N -1) - h_j_rule(state, state.N)) \
                                - state.F[state.N-1] * h_j_rule(state, (state.N -1)))
    new_state = state._replace(BE= BE_new,
                               DE= DE_new)
    return new_state

def initiate_solve_energy_balances(state: State):
    state = solve_energy_balances_condenser(state)
    state = solve_energy_balances_stages(state)
    state = solve_energy_balances_reboiler(state)
    return state
    # condenser = solve_energy_balances_condenser(state)
    # stages = solve_energy_balances_stages(state)
    # reboiler = solve_energy_balances_reboiler(state)
    # return condenser, stages, reboiler

def initiate_energy_balance_matrix(state: State):
    state = initiate_solve_energy_balances(state)
    rows, cols = jnp.diag_indices(len(jnp.zeros(N_max)))
    UR = rows[:-1]
    UC = cols[1:]
    diagonal = state.BE[1:]
    upper = state.CE[1:-1]
    new_state = state._replace(EE= state.EE.at[rows,cols].set(diagonal).at[UR, UC].set(upper))
    return new_state

def solve_energy_balances_vapor(state: State):
    state = solve_energy_balances_stream_copy(state)
    state = initiate_energy_balance_matrix(state)
    new_V = linalg.solve(state.EE[:state.N,:state.N],state.DE[1:state.N+1])
    new_state = state._replace(V= state.V.at[1:state.N+1].set(new_V))
    return new_state

def solve_energy_balances_liquid(state: State):
    new_L = state.RR * state.Distillate
    L_sum_values = [state.F[i] for i in range(2, (state.N + 1))]
    L_sum_array = jnp.array(L_sum_values)
    L_sum = jnp.sum(L_sum_array)
    new_L2 = jnp.array([state.V[i + 1] - state.Distillate + L_sum for i in range(1, state.N)])
    new_state = state._replace(L= state.L.at[0].set(new_L).at[1:state.N].set(new_L2))
    return new_state


def flow_rates_converged(state: State):
    return is_below_relative_error(state, state.L, state.L_old), is_below_relative_error(state, state.V[1:], state.V[1:])

def is_below_relative_error(state: State, new, old):
    diff = new - old
    rel_diff = diff / new
    max_rel_diff = jnp.abs(rel_diff).max()
    result = max_rel_diff < flow_rate_tol
    return result

def residual(x,p,T):
    K_vals = jnp.array([eval_SI_Depriester(T, p, c) for c in components])
    result = jnp.sum(x * K_vals) - 1
    return result

def custom_root(residual, x0, solver, tangent_solve):
    def cond_fun(args):
        x, fx, fdx = args
        return jnp.abs(fx) > 1e-6

    def body_fun(args):
        x, fx, fdx = args
        dx = solver(fx, fdx)
        x = x - dx
        fx, fdx = tangent_solve(x)
        return x, fx, fdx

    fx, fdx = tangent_solve(x0)
    x, _, _ = lax.while_loop(cond_fun, body_fun, (x0, fx, fdx))
    return x

def solver(fx, fdx):
    result = fx/fdx
    return result

def bubble_point(x,p,T_guess):
    def tangent_solve(T):
        fx = residual(x, p, T)
        fdx = grad(residual,argnums=2)(x, p, T)
        return fx, fdx

    root = custom_root(lambda T: residual(x,p,T), T_guess, solver, tangent_solve)
    return root

# print(bubble_point(jnp.array([0.2,0.3,0.5]),state.P_feed,jnp.array(300.0)))
# def bubble_point(x, K, p, T_guess):
#     """
#
#     :param x: mole fractions in liquid
#     :param p: total pressure
#     :param K: functions calculating K for each component
#     :param T_guess: guess temperature
#     :return: Temperature at which the liquid mixture begins to boil.
#     """
#     from scipy.optimize import root_scalar
#     sol = root_scalar(residual, args=(x, K, p), x0=T_guess-10, x1=T_guess+10)
#     return sol.root

def mass_balance_matrix(state: State):
    rows, cols = jnp.diag_indices(len(state.B))
    LR = rows[1:]
    LC = cols[:-1]
    UR = rows[:-1]
    UC = cols[1:]
    # e = e.at[LR, LC].set(lower)
    # e = e.at[UR, UC].set(upper)
    # e = e.at[rows, cols].set(diag)
    new_state = state._replace(E= state.E.at[LR, LC].set(state.A).at[UR, UC].set(state.C).at[rows, cols].set(state.B))
    return new_state

def solve_diagonal(state: State):
    state = mass_balance_matrix(state)
    result = linalg.solve(state.E[:state.N+1,:state.N+1], state.D[:state.N+1])
    return result

# make_ABC(state, 'n-Butane')
# print(solve_diagonal(state))

def update_T(state: State):
    T_update = jnp.array([bubble_T(state, i) for i in range(0, state.N+1)])
    new_state = state._replace(T= state.T.at[:state.N+1].set(T_update))
    return new_state

def tutorial(state: State):
    def initialize(state: State):
        state = state._replace(T_feed=bubble_T_feed(state))
        state = state._replace(T=state.T.at[:state.N + 1].set(state.T_feed))
        state = initialize_flow_rates(state)
        return state

    def mass_balance(state: State):
        state = solve_component_mass_bal(state, components[0])
        state = solve_component_mass_bal(state, components[1])
        state = solve_component_mass_bal(state, components[2])
        return state

    def loop_cond_mass_balance(state: State):
        condition = jnp.logical_not(T_is_converged(state))
        return condition

    def loop_body_mass_balance(state: State):
        state = update_K_values(state)
        state = mass_balance(state)
        state = update_T(state)
        return state

    def main_loop_mass_balance(state: State):
        final_state = lax.while_loop(loop_cond_mass_balance, loop_body_mass_balance, state)
        return final_state

    state = initialize(state)
    state = update_K_values(state)
    state = mass_balance(state)
    state = update_T(state)
    # state = main_loop_mass_balance(state)
    state = solve_energy_balances_vapor(state)
    state = solve_energy_balances_liquid(state)

    return state

print(tutorial(state))

# result = jax.jit(tutorial)(state)
# print(result)
    # # print(state.T_feed)
    #
    # # print(state.T_feed)
    #
    # # print(state.L)
    # # print(state.V)
    # state = update_K_values(state)
    # # print(state.K)
    #
    # state = solve_component_mass_bal(state, components[0])
    # state = solve_component_mass_bal(state, components[1])
    # state = solve_component_mass_bal(state, components[2])
    # print(state)
    # # print(state.l)
    # state = update_T(state)
    # # print(state.T)
    # # print(T_is_converged(state))
    # # iter = 0
    # # while not T_is_converged(state):
    # #     state = update_K_values(state)
    # #     for i in components:
    # #         solve_component_mass_bal(state, i)
    # #     state = update_T(state)
    # #     print(iter, state.T)
    # #     iter += 1



# class State(NamedTuple):
#     temperature: chex.Array
#     pressure: chex.Array
#     features: chex.Array
#
# def example_function(state: State, temperature_change: chex.Array) -> State:
#     new_state = State(
#         temperature=state.temperature + temperature_change,
#         pressure=state.pressure,
#         features=state.features)
#     new_state = state._replace(temperature = state.temperature + temperature_change)
#     return new_state
#
# if __name__ == '__main__':
#     state = State(temperature=jnp.array(1.0), pressure=jnp.array(2.0), features=jnp.array(3.0))
#     new_state = jax.jit(example_function)(state, jnp.array(1.0))
#     print(new_state)