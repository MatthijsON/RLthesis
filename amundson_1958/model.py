from typing import NamedTuple
from jumanji import ROOT_DIR, os

import chex
import jax.numpy as jnp
import jax
from jax.scipy import linalg
from jax import lax, grad
import matplotlib.pyplot as plt
import timeit
from jax import config
from jumanji.distillation.amundson_1958.types import State
config.update("jax_enable_x64", True)

'This work is based on the python based code to model a distillation column from DeJacko' \
'His work is based on the principles in Amundson 1958 where the iterative process of solving a distillation column is described'

'The way this Distillation column works is the following' \
'1. Specify your input variables: column pressure, feed flow, feed components, reflux ratio, number of stages, feed stage, mole fraction of feed components' \
'2. Initial values based on your input variables are made for the Temperature, vapor flow and liquid flow per stage.'

'3. K-values are evaluated based on the initial values' \
'4. The mass balance is evaluated for the initial values' \
'5. The mass balance determines the liquid per stage per component, which is linked back to mole fraction of the total mixture' \
'6. With these mole fractions the bubble point per stage is calculated' \
'7. Steps 3 to 6 are repeated in a while loop to converge the mass balance. This is done until the difference between the last iteration and current iteration is below temperature_tol' \

'8. The energy balance is solved based on the results from step 7' \
'9. The solved energy balance gives us new values for the liquid and vapor streams per stage' \
'10. To converge the energy balance steps 7 to 9 are repeated until the difference of the vapor and liquid values between iterations is below flow_rate_tol' \
'11. Plots are generated at the end showing the mole fraction per component per stage and the temperature profile throughout the column'


'Constants'
components = ['n-Butane', 'n-Pentane', 'n-Octane']
flow_rate_tol = jnp.array(1.0e-4)
temperature_tol = jnp.array(1.0e-2)
dampening_factor = jnp.array(1.0)
N_max = 100
z_feed1 = jnp.array([0.2, 0.5, 0.3])
num_constants = 5

'Component constants'
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

'Functions for assigning component constants'

path_depriester = os.path.join(ROOT_DIR, 'distillation', 'equilibrium_data', 'depriester.csv')
path_CpL = os.path.join(ROOT_DIR, 'distillation', 'equilibrium_data', 'heat_capacity_liquid.csv')
path_dH = os.path.join(ROOT_DIR, 'distillation', 'equilibrium_data', 'heats_of_vaporization.csv')

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

'State definition'
# class State(NamedTuple):
#     F_feed: chex.Array
#     P_feed: chex.Array
#     z_feed: dict
#     RR: chex.Array
#     Distillate: chex.Array
#     N: chex.Array
#     feed_stage: chex.Array
#     T_feed_guess: chex.Array
#     T_feed: chex.Array
#     L: chex.Array
#     V: chex.Array
#     L_old: chex.Array
#     V_old: chex.Array
#     F: chex.Array
#     z: dict
#     l: dict
#     T: chex.Array
#     T_old: chex.Array
#     K: dict
#     bubble_point: chex.Array
#     eps: chex.Array
#     A: chex.Array
#     B: chex.Array
#     C: chex.Array
#     D: chex.Array
#     E: chex.Array
#     BE: chex.Array
#     CE: chex.Array
#     DE: chex.Array
#     EE: chex.Array

'Assigning array length/values to the state (initial values)'

# state = State(
#     F_feed= jnp.array(1000.0),
#     P_feed= jnp.array(2.0*1e5),
#     z_feed= {key: val for key, val in zip(components, z_feed1)},
#     RR= jnp.array(1.0),
#     Distillate= jnp.array(400.0),
#     N= jnp.array(30),
#     feed_stage= jnp.array(15),
#     T_feed_guess= jnp.array(300.0),
#     T_feed= jnp.zeros(1),
#     L= jnp.zeros(N_max+1),
#     V= jnp.zeros(N_max+1),
#     L_old= jnp.zeros(N_max+1),
#     V_old= jnp.zeros(N_max+1),
#     F= jnp.zeros(N_max+1),
#     z= {key: jnp.zeros(N_max+1) for key in components},
#     l= {key: jnp.zeros(N_max+1) for key in components},
#     T= jnp.zeros(N_max+1),
#     T_old= jnp.zeros(N_max+1),
#     K= {key: jnp.zeros(N_max+1) for key in components},
#     bubble_point= jnp.zeros(N_max+1),
#     eps= jnp.zeros(N_max+1),
#     A= -1 * jnp.ones(N_max),
#     B= jnp.zeros(N_max+1),
#     C= jnp.zeros(N_max),
#     D= jnp.zeros(N_max+1),
#     E= jnp.zeros((N_max +1)**2).reshape(N_max+1,N_max+1),
#     BE= jnp.zeros(N_max+1),
#     CE= jnp.zeros(N_max+1),
#     DE= jnp.zeros(N_max+1),
#     EE= jnp.zeros((N_max)**2).reshape(N_max,N_max)
# )

'replacements'

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
    constants = jnp.array(CpL_func[c])
    def power_law_eval(i):
        result = constants[i-1] * jax.lax.pow(T, (i - 1.0))
        return result
    vmap = jax.vmap(power_law_eval)(jnp.arange(1,num_constants+1))
    result = jnp.sum(vmap)
    return result

def integral_CpL(T, c):
    constants = jnp.array(CpL_func[c])
    def power_law_int(i):
        result = constants[i-1] * jax.lax.pow(T, (i - 0.0))/i
        return result
    vmap = jax.vmap(power_law_int)(jnp.arange(1,num_constants+1))
    result = jnp.sum(vmap)
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
    rule_result = [state.z[c][stage] * h_pure_rule(c, state.T_feed) for c in components]
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
    return result
def y_ij_expr(state: State, i, j):
    evalSI = eval_SI_Depriester(state.T[j], state.P_feed, i)
    result = evalSI * x_ij_expr(state, i, j)
    return result

def Q_condenser_rule(state: State):
    return state.Distillate * (1 + state.RR) * (h_j_rule(state, 0) - H_j_rule(state, 1))

def Q_reboiler_rule(state: State):
    return state.Distillate * h_j_rule(state, 0) + (state.F_feed - state.Distillate) * h_j_rule(state, state.N) \
            - state.F_feed * h_feed_rule(state, state.feed_stage) - Q_condenser_rule(state)

'Updating the K-values per component'

def update_K(state: State, component):
    mask = jnp.arange(N_max+1) < state.N+1
    new_K_component = state.K[component].copy()
    new_K_component = jnp.where(mask, eval_SI_Depriester(state.T, state.P_feed, component), new_K_component)

    return new_K_component

def update_K_values(state: State):
    new_K = {component: update_K(state, component) for component in components}
    mask = jnp.arange(N_max+1) < state.N+1
    new_T = jnp.where(mask, state.T, 0)
    new_state = state._replace(K= new_K,
                               T_old= new_T
                               )
    return new_state

'Bubble point calculation functions'

def update_T_values(state: State):
    def T_values(i):
        result = state.T_old[i] + dampening_factor * bubble_T(state, i) - state.T_old[i]
        return result
    T_test = jax.vmap(T_values)(jnp.arange(N_max+1))

    mask = jnp.arange(N_max+1) < state.N+1
    new_T = jnp.where(mask, T_test, state.T_old)

    new_state = state._replace(T= new_T)

    return new_state

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

def bubble_T(state: State, stage):
    l_component = [state.l[c][stage] for c in components]
    l_component_array = jnp.array(l_component)
    l_total = jnp.sum(l_component_array)
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
    mask = jnp.arange(N_max+1) < state.N+1
    result = jnp.where(mask, state.T_feed, state.T)
    return result

def bubble_T_feed(state: State):
    x_vals = jnp.array([state.z_feed[c] for c in components])
    T_bubble = bubble_point(x_vals, state.P_feed, state.T_feed_guess)
    return T_bubble

def initialize_flow_rates(state: State):
    new_L_above_feed = state.RR * state.Distillate
    new_L_under_feed = state.RR * state.Distillate + state.F_feed
    new_L_at_N = state.F_feed - state.Distillate

    mask1 = jnp.arange(N_max+1) < state.feed_stage
    mask2 = jnp.logical_and(jnp.arange(N_max+1) >= state.feed_stage, jnp.arange(N_max+1) < state.N)
    mask3 = jnp.arange(N_max+1) == state.N
    mask4 = jnp.logical_and(jnp.arange(N_max+1) > 0, jnp.arange(N_max+1) < state.N+1)

    new_L = jnp.where(mask1, new_L_above_feed, state.L)
    new_L = jnp.where(mask2, new_L_under_feed, new_L)
    new_L = jnp.where(mask3, new_L_at_N, new_L)

    new_V_values = state.RR * state.Distillate + state.Distillate
    new_V = jnp.where(mask4, new_V_values, state.V)

    new_state = state._replace(V= new_V,
                               L= new_L)
    return new_state

def T_is_converged(state: State):
    eps = jnp.abs(state.T - state.T_old)
    return eps.max() < temperature_tol

'Mass balance functions'

def make_ABC(state: State, component):
    # assert abs(state.V[0]) < 1e-8         'Doesn't work in JAX/JIT, doesn't seem to have a good alternative, but its only a check'
    B_condenser = 1 + state.Distillate / state.L[0]
    C_condenser = -state.V[1] * state.K[component][1] / state.L[1]
    D_condenser = state.F[0] * state.z[component][0]

    Bottoms = state.F_feed - state.Distillate
    B_reboiler = 1.0 + state.V[state.N] * state.K[component][state.N] / Bottoms
    D_reboiler = state.F[state.N] * state.z[component][state.N]

    B_values = 1 + state.V * state.K[component] / state.L
    C_values = -state.V[1:] * state.K[component][1:] / state.L[1:]
    D_values = state.F * state.z[component]
    'note: above values differ from original stage calculations, check back here if anything goes wrong later'

    mask = jnp.logical_and(jnp.arange(N_max+1) > 0, jnp.arange(N_max+1) < state.N)
    mask2 = jnp.logical_and(jnp.arange(N_max) > 0, jnp.arange(N_max) < state.N)

    new_B = jnp.where(mask, B_values, 1)
    new_B = new_B.at[0].set(B_condenser)
    new_B = new_B.at[state.N].set(B_reboiler)
    new_B = new_B.at[state.N+2].set(0)

    new_C = jnp.where(mask2, C_values, 1)
    new_C = new_C.at[0].set(C_condenser)
    new_C = new_C.at[state.N+2].set(0)

    new_D = jnp.where(mask, D_values, 0)
    new_D = new_D.at[0].set(D_condenser)
    new_D = new_D.at[state.N].set(D_reboiler)

    new_state = state._replace(B = new_B,
                               C = new_C,
                               D = new_D)
    return new_state

def mass_balance_matrix(state: State):
    rows, cols = jnp.diag_indices(N_max+1)
    LR = rows[1:]
    LC = cols[:-1]
    UR = rows[:-1]
    UC = cols[1:]
    new_state = state._replace(E= state.E.at[LR, LC].set(state.A).at[UR, UC].set(state.C).at[rows, cols].set(state.B))
    return new_state

def solve_diagonal(state: State):
    state = mass_balance_matrix(state)
    result = linalg.solve(state.E,state.D)
    return result

def solve_component_mass_bal(state: State, component):
    state = make_ABC(state, component)
    mask = jnp.arange(N_max+1) < state.N+1
    new_l = state.l.copy()
    new_l[component] = jnp.where(mask, solve_diagonal(state), new_l[component])
    new_state = state._replace(l= new_l)
    return new_state

# def update_flow_rates(state: State):
#     l_values = jnp.stack([state.l[c] for c in components])
#     L_values = jnp.sum(l_values, axis=0)
#     new_V_condenser = (state.RR + 1) * state.Distillate
#     F_values = [state.F[i] for i in range(2, (state.N+1))]
#     F_values_array = jnp.array(F_values)
#
#     mask = jnp.logical_and(jnp.arange(N_max+1) > 1, jnp.arange(N_max+1) < state.N+1)
#     F_values_test = jnp.where(mask, state.F, 0.0)
#     F_test_sum = jnp.sum(F_values_test)
#     F_sum = jnp.sum(F_values_array)
#     mask2 = jnp.logical_and(jnp.arange(N_max+1) > 1, jnp.arange(N_max+1) < state.N+1)
#     V_values_test = jnp.where(mask2, L_values + state.Distillate - F_test_sum, 0.0)
#     V_values = [state.L[i-1] + state.Distillate - F_sum for i in range(2, (state.N+1))]
#     V_other = jnp.array(V_values)
#     new_state = state._replace(L= L_values,
#                                V= state.V.at[0].set(0).at[1].set(new_V_condenser).at[2:state.N+1].set(V_other)
#                                )
#     return new_state

'Energy balance functions'

def solve_energy_balances_stream_copy(state: State):
    new_state = state._replace(L_old= state.L_old.at[:].set(state.L[:]),
                               V_old= state.V_old.at[:].set(state.V[:])
                               )
    return new_state

def solve_energy_balances_condenser(state: State):
    new_CE = h_j_rule(state, 0) - H_j_rule(state, 1)
    new_DE = state.F[0] * h_feed_rule(state, 0) + Q_condenser_rule(state)
    new_state = state._replace(BE= state.BE.at[0].set(0),
                               CE= state.CE.at[0].set(new_CE),
                               DE= state.DE.at[0].set(new_DE)
                               )
    return new_state

def solve_energy_balances_stages(state: State):
    mask = jnp.logical_and(jnp.arange(N_max+1) > 0, jnp.arange(N_max+1) < state.N)
    mask2 = jnp.logical_and(jnp.arange(N_max+1) > 1, jnp.arange(N_max+1) < state.N+1)
    mask3 = jnp.arange(N_max+1) < (state.N-1)

    def BE_function(i):
        result = H_j_rule(state, i) - h_j_rule(state, (i-1))
        return result

    BE_values = jax.vmap(BE_function)(jnp.arange(N_max+1))
    new_BE = jnp.where(mask, BE_values, 1)
    # new_BE = new_BE.at[state.N+1].set(0)

    def CE_function(i):
        result = h_j_rule(state, i) - H_j_rule(state, (i+1))
        return result

    CE_values = jax.vmap(CE_function)(jnp.arange(N_max+1))
    new_CE = jnp.where(mask, CE_values, 1)
    # new_CE = new_CE.at[state.N+1].set(0)

    def F_function(i):
        result = state.F[i]
        return result

    def rule_function(i):
        result = h_j_rule(state, i)
        return result

    vmap_F = jax.vmap(F_function)(jnp.arange(N_max+1))
    vmap_rule = jax.vmap(rule_function)(jnp.arange(N_max+1))

    sum1_vmap_F = jnp.where(mask2, vmap_F, 0)
    sum1_vmap_rule = jnp.where(mask, vmap_rule, 0)
    sum1_value = jnp.sum(sum1_vmap_F[2:] * sum1_vmap_rule[1:-1])

    sum2_vmap_F = jnp.where(mask, vmap_F, 0)
    sum2_vmap_rule = jnp.where(mask3, vmap_rule, 0)
    sum2_value = jnp.sum(sum2_vmap_F[1:] * sum2_vmap_rule[:-1])

    def final_function_below_feed(i):
        result = state.F[i] * h_feed_rule(state,i) - state.Distillate * (h_j_rule(state, (i-1)) - h_j_rule(state, i)) - sum1_value + sum2_value
        return result

    def final_function_above_feed(i):
        result = state.F[i] * h_feed_rule(state,i) - state.Distillate * (h_j_rule(state, (i-1)) - h_j_rule(state, i)) - (jnp.sum(sum1_vmap_F) * h_j_rule(state, i)) + (jnp.sum(sum2_vmap_F) * h_j_rule(state, i-1))
        return result

    final_values_below_feed = jax.vmap(final_function_below_feed)(jnp.arange(N_max+1))
    final_values_above_feed = jax.vmap(final_function_above_feed)(jnp.arange(N_max+1))
    mask4 = jnp.logical_and(jnp.arange(N_max+1) > 0, jnp.arange(N_max+1) < state.feed_stage)
    mask5 = jnp.logical_and(jnp.arange(N_max+1) >= state.feed_stage, jnp.arange(N_max+1) < state.N+1)
    new_DE = jnp.where(mask4, final_values_below_feed,0)
    new_DE = jnp.where(mask5, final_values_above_feed, new_DE)
    feed_stage_value = state.F[state.feed_stage] * h_feed_rule(state, state.feed_stage) - state.Distillate * (h_j_rule(state, state.feed_stage-1) - h_j_rule(state, state.feed_stage)) - (jnp.sum(sum1_vmap_F) * h_j_rule(state, state.feed_stage))
    new_DE = new_DE.at[state.feed_stage].set(feed_stage_value)
    # new_DE = new_DE.at[state.N+1].set(0)

    new_state = state._replace(BE= new_BE,
                               CE= new_CE,
                               DE= new_DE)
    return new_state

def solve_energy_balances_reboiler(state: State):
    new_BE = state.BE.at[state.N].set(H_j_rule(state, state.N) - h_j_rule(state, (state.N - 1)))
    new_DE = state.DE.at[state.N].set(state.F[state.N] * h_feed_rule(state, state.N) + Q_reboiler_rule(state) \
                                - (state.F_feed - state.Distillate) * (h_j_rule(state, state.N -1) - h_j_rule(state, state.N)) \
                                - state.F[state.N-1] * h_j_rule(state, (state.N-1)))
    new_state = state._replace(BE= new_BE,
                               DE= new_DE)
    return new_state

def initiate_solve_energy_balances(state: State):
    state = solve_energy_balances_stages(state)
    state = solve_energy_balances_condenser(state)
    state = solve_energy_balances_reboiler(state)
    return state

def initiate_energy_balance_matrix(state: State):
    rows, cols = jnp.diag_indices(N_max)
    UR = rows[:-1]
    UC = cols[1:]
    diagonal = state.BE[1:]
    upper = state.CE[1:-1]
    new_state = state._replace(EE= state.EE.at[rows,cols].set(diagonal).at[UR, UC].set(upper))
    return new_state

def solve_energy_balances_vapor(state: State):
    state = solve_energy_balances_stream_copy(state)
    state = initiate_solve_energy_balances(state)
    state = initiate_energy_balance_matrix(state)
    new_V_values_test = linalg.solve(state.EE,state.DE[1:])
    mask = jnp.logical_and(jnp.arange(N_max) >= 0, jnp.arange(N_max) < state.N)
    new_V_test = jnp.where(mask, new_V_values_test, 1)
    # new_V_test = jnp.where(mask, new_V_values_test, state.V[1:])
    new_V = state.V.at[1:].set(new_V_test)
    new_state = state._replace(V= new_V)
    return new_state

def solve_energy_balances_liquid(state: State):
    L_top = state.RR * state.Distillate
    L_bottom = state.F_feed - state.Distillate

    def F_function(i):
        result = state.F[i]
        return result

    vmap_F = jax.vmap(F_function)(jnp.arange(N_max+1))
    mask = jnp.logical_and(jnp.arange(N_max+1) >= 2, jnp.arange(N_max+1) < state.N+1)
    F_values = jnp.where(mask, vmap_F, 0)
    F_sum = jnp.sum(F_values)

    def rule_function_above_feed(i):
        result = state.V[i+1] - state.Distillate
        return result

    def rule_function_below_feed(i):
        result = state.V[i+1] - state.Distillate + F_sum
        return result

    L_values_below_feed = jax.vmap(rule_function_below_feed)(jnp.arange(N_max+1))
    L_values_above_feed = jax.vmap(rule_function_above_feed)(jnp.arange(N_max+1))
    mask2 = jnp.logical_and(jnp.arange(N_max+1) >= 0, jnp.arange(N_max+1) < state.N+1)
    mask3 = jnp.logical_and(jnp.arange(N_max + 1) > 0, jnp.arange(N_max + 1) < state.feed_stage)
    mask4 = jnp.logical_and(jnp.arange(N_max + 1) >= state.feed_stage, jnp.arange(N_max + 1) < state.N)
    new_L = jnp.where(mask3, L_values_above_feed, 0)
    new_L = jnp.where(mask4, L_values_below_feed, new_L)
    new_L = new_L.at[0].set(L_top).at[state.N].set(L_bottom)
    new_L = jnp.where(mask2, new_L, 1.0)
    new_state = state._replace(L = new_L)
    return new_state

'Check functions for the loops'

def flow_rates_converged_liquid(state: State):
    result = is_below_relative_error_liquid(state)
    return result

def flow_rates_converged_vapor(state: State):
    result = is_below_relative_error_vapor(state)
    return result

def is_below_relative_error_liquid(state: State):
    diff = state.L - state.L_old
    rel_diff = diff / state.L
    max_rel_diff = jnp.abs(rel_diff).max()
    result = max_rel_diff < flow_rate_tol
    return result

def is_below_relative_error_vapor(state: State):
    diff = state.V[1:] - state.V_old[1:]
    rel_diff = diff / state.V[1:]
    max_rel_diff = jnp.abs(rel_diff).max()
    result = max_rel_diff < flow_rate_tol
    return result

def update_T(state: State):
    def T_values(i):
        result = bubble_T(state, i)
        return result
    T_update_values = jax.vmap(T_values)(jnp.arange(N_max+1))
    mask = jnp.arange(N_max+1) < state.N+1
    T_update = jnp.where(mask, T_update_values, 0)
    new_state = state._replace(T= T_update)
    return new_state

'Tutorial function, executing the entire code'

def tutorial(state: State):
    def initialize(state: State):
        new_z = {component: state.z[component].at[state.feed_stage].set(state.z_feed[component]) for component in
                 components}
        state = state._replace(z=new_z)
        state = state._replace(F=state.F.at[state.feed_stage].set(state.F_feed))
        state = state._replace(T_feed=bubble_T_feed(state))
        mask = jnp.arange(N_max+1) < state.N+1
        new_T = jnp.where(mask, state.T_feed, state.T)
        state = state._replace(T=new_T)
        state = initialize_flow_rates(state)
        return state

    def mass_balance(state: State):
        # def solve_mb(c):
        #     result = solve_component_mass_bal(state, c)
        #     return result
        #
        # state = jax.vmap(solve_mb)()

        "Raises 'string can not be passed to this function in Vmap'"

        state = solve_component_mass_bal(state, components[0])
        state = solve_component_mass_bal(state, components[1])
        state = solve_component_mass_bal(state, components[2])
        return state

    def loop_cond_mass_balance(state: State):
        condition = jnp.logical_not(T_is_converged(state))
        # condition = T_is_converged(state)
        return condition

    def loop_body_mass_balance(state: State):
        state = update_K_values(state)
        state = mass_balance(state)
        state = update_T(state)
        return state

    def main_loop_mass_balance(state: State):
        final_state = lax.while_loop(loop_cond_mass_balance, loop_body_mass_balance, state)
        return final_state

    def loop_cond_energy_balance(state: State):
        condition = jnp.logical_and(jnp.logical_not(flow_rates_converged_liquid(state)),jnp.logical_not(flow_rates_converged_vapor(state)))
        # condition = jnp.logical_and(flow_rates_converged_liquid(state),flow_rates_converged_vapor(state))
        return condition

    def loop_body_energy_balance(state):
        state = mass_balance(state)
        state = update_T(state)
        state = main_loop_mass_balance(state)
        state = solve_energy_balances_vapor(state)
        state = solve_energy_balances_liquid(state)
        return state

    def main_loop_energy_balance(state: State):
        final_state = lax.while_loop(loop_cond_energy_balance, loop_body_energy_balance, state)
        return final_state

    state = initialize(state)
    state = update_K_values(state)
    state = mass_balance(state)
    state = update_T(state)

    state = main_loop_mass_balance(state)

    state = solve_energy_balances_vapor(state)
    state = solve_energy_balances_liquid(state)

    state = mass_balance(state)
    state = update_T(state)
    state = main_loop_mass_balance(state)
    state = solve_energy_balances_vapor(state)
    state = solve_energy_balances_liquid(state)

    state = main_loop_energy_balance(state)
    "Original while loop mass balance, how it should look like"
    # while not T_is_converged(state):
    #     start_loop += 1
    #     state = update_K_values(state)
    #     state = mass_balance(state)
    #     state = update_T(state)
    #     print(start_loop)

    "Original while loop energy balance, how it should look like"

    # while not (flow_rates_converged_vapor(state) and flow_rates_converged_liquid(state)):
    #     outer_loop += 1
    #     state = mass_balance(state)
    #     state = update_T(state)
    #     while not T_is_converged(state):
    #         inner_loop += 1
    #         state = update_K_values(state)
    #         state = mass_balance(state)
    #         state = update_T(state)
    #         print(inner_loop)
    #     state = solve_energy_balances_vapor(state)
    #     state = solve_energy_balances_liquid(state)
    #     print(outer_loop, inner_loop, state.V)

    return state

# start = timeit.default_timer()
# state_results = (tutorial)(state)
# stop = timeit.default_timer() - start
# print(state_results)
# print(stop)
# def generate_plots(state_results):
#     x = jnp.array([state_results.l[c][:]/state_results.L[:] for c in components])
#
#     fig1 = plt.figure(1)
#     ax = fig1.add_subplot(111)
#     ax.plot((jnp.arange(state_results.N+1)), state_results.T[:state_results.N+1], 'o')
#     ax.set_xlabel('Stage Number')
#     ax.set_ylabel('Temperature [K]')
#
#     fig2 = plt.figure(2)
#     ax2 = fig2.add_subplot(111)
#     ax2.plot((jnp.arange(state_results.N+1)), x[0][:state_results.N+1], label='n-Butane')
#     ax2.plot((jnp.arange(state_results.N+1)), x[1][:state_results.N+1], label='n-Pentane')
#     ax2.plot((jnp.arange(state_results.N+1)), x[2][:state_results.N+1], label='n-Octane')
#     ax2.set_ylabel('Liquid phase mole fraction')
#     ax2.set_xlabel('Stage Number')
#     ax2.legend()
#     plt.show()
#     return fig1, fig2
#
# fig1, fig2 = generate_plots(state_results)
#
#
# def jit_compile(state, n_times):
#     start = timeit.default_timer()
#     for i in range(n_times):
#         jax.tree_util.tree_flatten(jax.jit(tutorial)(state)[0].block_until_ready())
#         # jax.jit(tutorial)(state)
#     time_req = timeit.default_timer() - start
#     return time_req
#
# def non_jit(state, n_times):
#     start = timeit.default_timer()
#     for i in range(n_times):
#         tutorial(state)
#     time_req1 = timeit.default_timer() - start
#     return time_req1
#
# def jit_run(state, n_times):
#     function = jax.jit(tutorial)
#     function(state)
#     start = timeit.default_timer()
#     for i in range(n_times):
#         jax.tree_util.tree_flatten(function(state)[0].block_until_ready())
#         # function(state)
#     time_req2 = timeit.default_timer() - start
#     return time_req2
#
# n_times_run = [1]
# times_jit_compile = []
# times_non_jit = []
# times_jit_run = []
#
# n_times_test = 2
#
# for i in range(n_times_test):
#     for j in n_times_run:
#         run_jit_compile = jit_compile(state,j)
#         times_jit_compile.append(run_jit_compile)
#
#         # run_non_jit = non_jit(state,j)
#         # times_non_jit.append(run_non_jit)
#
#         run_jit_run = jit_run(state,j)
#         times_jit_run.append(run_jit_run)
#
# print(times_jit_compile)
# print(times_non_jit)
# print(times_jit_run)
#
# total_jit = jnp.sum(jnp.array(times_jit_run))
# print(total_jit)
#
# times_jit_compile = jnp.asarray(times_jit_compile)
# avg_time_jit_compile = jnp.mean(times_jit_compile)
# std_time_jit_compile = jnp.std(times_jit_compile)
#
# times_non_jit = jnp.asarray(times_non_jit)
# avg_time_non_jit = jnp.mean(times_non_jit)
# std_time_non_jit = jnp.std(times_non_jit)
#
# times_jit_run = jnp.asarray(times_jit_run)
# avg_time_jit_run = jnp.mean(times_jit_run)
# std_time_jit_run = jnp.std(times_jit_run)

# fig, ax = plt.subplots()
# plt.plot(range(n_times_test), times_non_jit, '-o', alpha=0.5, color='#E182FD')
# plt.plot(range(n_times_test), times_jit_compile, '-o', alpha=0.5, color='#6E98F8')
# plt.plot(range(n_times_test), times_jit_run, '-o', alpha=0.5, color='#4BA89B')
# ax.set_yscale('log')
# plt.ylabel("Computational time (s)", fontsize=11)
# plt.legend(["run time (no compilation)", "compile and run time", "run time (with compilation)"], loc="best")
# plt.xlabel("Run number", fontsize=11)
# plt.savefig("reactor.svg", format='svg', dpi=1200)
# plt.show()
#
# labels = ["A"]
# x_pos = jnp.arange(len(labels))
#
# fig, ax = plt.subplots()
# width = 0.25
# ax.bar(x_pos, avg_time_non_jit, width, align='center', alpha=0.5, ecolor='black', capsize=10,
#            color='#E182FD')
# ax.bar(x_pos + width, avg_time_jit_compile, width, align='center', alpha=0.5, ecolor='black', capsize=10,
#            color='#6E98F8')
# ax.bar(x_pos + 2*width, avg_time_jit_run, width, align='center', alpha=0.5, ecolor='black', capsize=10,
#            color='#4BA89B')
# ax.set_xticks(x_pos+width/2)
# labels = [" "]
# ax.set_xticklabels(labels)
# plt.legend(["run time (no compilation)", "compile and run time", "run time (with compilation)"], loc="best")
# ax.set_yscale('log')
# ax.set_ylabel("Computational time (s)", fontsize=11)
# plt.ylim([0, 100])
# plt.savefig("jitbargraph.svg", format='svg', dpi=1200)
# plt.show()