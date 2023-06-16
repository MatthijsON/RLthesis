from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

class PerCompoundProperty(NamedTuple):

    n_Butane: chex.Array
    n_Pentane: chex.Array
    n_Octane: chex.Array

class StreamSpecification(NamedTuple):

    temperature: chex.Array
    pressure: chex.Array
    molar_flows: PerCompoundProperty

class ColumnInputSpec(NamedTuple):

    n_stages: chex.Array
    feed_stage: chex.Array
    pressure: chex.Array
    RR: chex.Array

class ColumnOutputSpec(NamedTuple):

    vapor_flow_per_stage: chex.Array
    liquid_flow_per_stage: chex.Array
    temperature_per_stage: chex.Array

class PuritySpec(NamedTuple):

    purity: chex.Numeric

class Observation(NamedTuple):

    created_state: PerCompoundProperty
    max_value: PerCompoundProperty
    # input_spec: ColumnInputSpec
    # output_spec: ColumnOutputSpec
    step_count: chex.Numeric

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
    A: chex.Array
    B: chex.Array
    C: chex.Array
    D: chex.Array
    E: chex.Array
    BE: chex.Array
    CE: chex.Array
    DE: chex.Array
    EE: chex.Array
    molar_flow: PerCompoundProperty
    step_count: chex.Numeric
    key: chex.PRNGKey



