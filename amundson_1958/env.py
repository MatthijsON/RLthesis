from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
# import matplotlib
# import matplotlib.animation
# import matplotlib.artist

from jumanji import specs
from jumanji.env import Environment
from jumanji.distillation.amundson_1958.types import State, Observation, PerCompoundProperty, StreamSpecification, ColumnInputSpec, ColumnOutputSpec, PuritySpec
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.distillation.amundson_1958.model import tutorial

N_max = 100
components = ['n-Butane', 'n-Pentane', 'n-Octane']
z_feed1 = jnp.array([0.2, 0.5, 0.3])

class Distillation_Column(Environment[State]):

    def __init__(self,
                 time_limit: int = 10,
                 product_spec: PuritySpec = 0.8,
                 n_stages_bound: int = 100,
                 pressure_bound_low: int = 1,
                 pressure_bound_high: int = 10,
                 RR_bound_low: int = 1,
                 RR_bound_high: int = 5):

        self.time_limit = time_limit
        self.product_spec = product_spec
        self.n_stages_bound = n_stages_bound
        self.pressure_bound_low = pressure_bound_low
        self.pressure_bound_high = pressure_bound_high
        self.RR_bound_low = RR_bound_low
        self.RR_bound_high = RR_bound_high

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        key, stage_key, pressure_key, RR_key = jax.random.split(key, 4)

        N_reset = jax.random.randint(
            stage_key,
            shape=(1,),
            minval= jnp.array(2),
            maxval= jnp.array(self.n_stages_bound),
        )

        pressure_reset = jax.random.randint(
            pressure_key,
            shape=(1,),
            minval= jnp.array(self.pressure_bound_low),
            maxval= jnp.array(self.pressure_bound_high),
        )

        RR_reset = jax.random.randint(
            RR_key,
            shape=(1,),
            minval = jnp.array(self.RR_bound_low),
            maxval = jnp.array(self.RR_bound_high),
        )

        feed_stage_reset = jnp.array(jnp.round((N_reset/2)+0.5))

        state = State(
            F_feed=jnp.array(1000.0),
            P_feed=jnp.array(pressure_reset * 1e5),
            z_feed={key: val for key, val in zip(components, z_feed1)},
            RR=RR_reset,
            Distillate=jnp.array(400.0),
            N=N_reset,
            feed_stage=feed_stage_reset,
            T_feed_guess=jnp.array(300.0),
            T_feed=jnp.zeros(1),
            L=jnp.zeros(N_max + 1),
            V=jnp.zeros(N_max + 1),
            L_old=jnp.zeros(N_max + 1),
            V_old=jnp.zeros(N_max + 1),
            F=jnp.zeros(N_max + 1),
            z={key: jnp.zeros(N_max + 1) for key in components},
            l={key: jnp.zeros(N_max + 1) for key in components},
            T=jnp.zeros(N_max + 1),
            T_old=jnp.zeros(N_max + 1),
            K={key: jnp.zeros(N_max + 1) for key in components},
            A=-1 * jnp.ones(N_max),
            B=jnp.zeros(N_max + 1),
            C=jnp.zeros(N_max),
            D=jnp.zeros(N_max + 1),
            E=jnp.zeros((N_max + 1) ** 2).reshape(N_max + 1, N_max + 1),
            BE=jnp.zeros(N_max + 1),
            CE=jnp.zeros(N_max + 1),
            DE=jnp.zeros(N_max + 1),
            EE=jnp.zeros((N_max) ** 2).reshape(N_max, N_max),
            molar_flow= PerCompoundProperty(n_Butane=jnp.zeros(N_max+1), n_Pentane=jnp.zeros(N_max+1), n_Octane=jnp.zeros(N_max+1)),
            step_count= jnp.array(0, jnp.int32),
            key=key
        )

        timestep = restart(observation=self.state_to_observation(state))

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:

        "generate input variables from action"
        input_spec = self.generate_input_spec(action)
        "set input variables in the state"
        state = state._replace(N=input_spec.n_stages,
                               feed_stage=input_spec.feed_stage,
                               P_feed=input_spec.pressure,
                               RR=input_spec.RR)
        "calculate next state"
        next_state = jax.jit(tutorial)(state)

        step_count = state.step_count + 1
        molar_flows = PerCompoundProperty(
            n_Butane= next_state.l[components[0]][:] / next_state.L[:],
            n_Pentane= next_state.l[components[1]][:] / next_state.L[:],
            n_Octane= next_state.l[components[2]][:] / next_state.L[:]
        )
        next_state = next_state._replace(step_count=step_count,
                                         molar_flow=molar_flows)

        above_purity = jnp.array([molar_flows.n_Butane.max(), molar_flows.n_Pentane.max(), molar_flows.n_Octane.max()]).max() >= self.product_spec
        done = above_purity | (step_count >= self.time_limit)
        reward = jnp.array([molar_flows.n_Butane.max(), molar_flows.n_Pentane.max(), molar_flows.n_Octane.max()]).max()
        observation = self.state_to_observation(next_state)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation
        )

        return next_state, timestep

    def generate_input_spec(self, action: chex.Array) -> ColumnInputSpec:
        new_N = jnp.int32(jnp.interp(action[0], jnp.array([0,2]), jnp.array([0,self.n_stages_bound])) - action[0]*30 + 10)
        new_feed_stage = jnp.int32(jnp.round(new_N/2+0.5))
        new_pressure = jnp.interp(action[1], jnp.array([0,2]), jnp.array([self.pressure_bound_low,self.pressure_bound_high])) * 1e5
        new_RR = jnp.interp(action[2], jnp.array([0,2]), jnp.array([self.RR_bound_low, self.RR_bound_high]))

        return ColumnInputSpec(
            n_stages=new_N,
            feed_stage=new_feed_stage,
            pressure=new_pressure,
            RR=new_RR,
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        # shape = (N_max+1,)
        # individual = PerCompoundProperty(n_Butane= specs.BoundedArray(shape=shape,dtype=float,minimum=0,maximum=1,name="n-Butane"),
        #                                  n_Pentane= specs.BoundedArray(shape=shape,dtype=float,minimum=0,maximum=1,name="n-Pentane"),
        #                                  n_Octane= specs.BoundedArray(shape=shape,dtype=float,minimum=0,maximum=1,name="n-Octane")
        #                                  )

        created_state = specs.BoundedArray(
            shape=(PerCompoundProperty),
            dtype=float,
            minimum=0,
            maximum=1,
            name="created_state"
        )

        max_value = specs.BoundedArray(
            shape=(PerCompoundProperty),
            dtype=float,
            minimum=0,
            maximum=1,
            name="max_value"
        )

        step_count = specs.DiscreteArray(
            self.time_limit, dtype=jnp.int32, name="step_count"
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            created_state=created_state,
            max_value=max_value,
            step_count=step_count,
        )


    def action_spec(self) -> specs.MultiDiscreteArray:

        return specs.MultiDiscreteArray(jnp.array([3, 3, 3]), name="action")


    def state_to_observation(self, state: State) -> Observation:
        molar_flows = state.molar_flow
        max_value = PerCompoundProperty(n_Butane=molar_flows.n_Butane.max(), n_Pentane=molar_flows.n_Pentane.max(), n_Octane=molar_flows.n_Octane.max())
        step_count = state.step_count
        # input_spec = ColumnInputSpec(n_stages=state.N, feed_stage=state.feed_stage, pressure=state.P_feed, RR=state.RR)
        # output_spec = ColumnOutputSpec(vapor_flow_per_stage=state.V, liquid_flow_per_stage=state.L, temperature_per_stage=state.T)


        return Observation(
            created_state=molar_flows,
            max_value=max_value,
            # input_spec=input_spec,
            # output_spec=output_spec,
            step_count=step_count,
        )
