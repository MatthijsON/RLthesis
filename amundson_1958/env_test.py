import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
import py
import pytest

from jumanji.distillation.amundson_1958.env import Distillation_Column, State
from jumanji.distillation.amundson_1958.types import Observation, PerCompoundProperty, PuritySpec, ColumnInputSpec
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep

@pytest.fixture(scope="module")
def distillation_column() -> Distillation_Column:
    return Distillation_Column()

def test_reset(distillation_column: Distillation_Column) -> None:
    reset_fn = jax.jit(chex.assert_max_traces(distillation_column.reset, n=1))
    state1, timestep1 = reset_fn(jax.random.PRNGKey(1))
    state2, timestep2 = reset_fn(jax.random.PRNGKey(2))
    print(timestep1)
    print(timestep2)
    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step_count == 0
    assert jnp.all(state1.molar_flow.n_Butane) == 0
    assert_is_jax_array_tree(state1)
    assert state1.N != state2.N
    assert state1.P_feed != state2.P_feed
    assert state1.RR != state2.RR
    assert not jnp.array_equal(state1.key, state2.key)
    assert not jnp.array_equal(state1.key, state2.key)

def test_step(distillation_column: Distillation_Column) -> None:
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(distillation_column.step, n=1))
    state_key = jax.random.PRNGKey(0)
    state, timestep = distillation_column.reset(state_key)
    action = distillation_column.action_spec().generate_value()
    next_state, next_timestep = step_fn(state, action)
    print(timestep)
    print(next_timestep)
    print(next_state)
    assert not jnp.array_equal(next_state.N, state.N)
    assert next_state.step_count != state.step_count
    assert not jnp.array_equal(next_state.P_feed, state.P_feed)
    assert not jnp.array_equal(next_state.RR, state.RR)
    assert jnp.any(next_state.molar_flow.n_Butane) != 0

def test_column__does_not_smoke(distillation_column: Distillation_Column) -> None:
    check_env_does_not_smoke(distillation_column)

def test_column__no_nan(distillation_column: Distillation_Column) -> None:
    reset_fn = jax.jit(distillation_column.reset)
    step_fn = jax.jit(distillation_column.step)
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    action = distillation_column.action_spec().generate_value()
    next_state, next_timestep = step_fn(state, action)
    chex.assert_tree_all_finite((next_state, next_timestep))