import chex
import jax
import jax.numpy as jnp
from jumanji.distillation.amundson_1958.env import Distillation_Column, State

# def distillation_column() -> Distillation_Column:
#     return Distillation_Column()
#
#
# state_key, action_key = jax.random.split(jax.random.PRNGKey(0))
# # action1, action2 = jax.random.choice(
# #             action_key,
# #             jnp.arange(distillation_column().action_spec()._num_values),
# #             shape=(6,),
# #             replace=False
# #         )
# action1 = distillation_column().action_spec().generate_value()
# print(action1)
# action2 = distillation_column().action_spec().generate_value()
# print(action2)

action = jnp.array([0,0,0])
RR_bound_low = 1
RR_bound_high = 5
test = jnp.interp(action[2], jnp.array([0,2]), jnp.array([RR_bound_low, RR_bound_high]))
print(test)

from jumanji.distillation.amundson_1958.types import PerCompoundProperty
from jumanji.specs import BoundedArray

test1 = jnp.arange(5) + 1
test2 = jnp.arange(5) + 2
test3 = jnp.arange(5)

print(test1, test2, test3)

molar_flows = PerCompoundProperty(n_Butane=test1,n_Pentane=test2,n_Octane=test3)
test5 = jnp.shape(PerCompoundProperty)
max_value = PerCompoundProperty(n_Butane=jnp.array(molar_flows.n_Butane.max()), n_Pentane=jnp.array(molar_flows.n_Pentane.max()), n_Octane=jnp.array(molar_flows.n_Octane.max()))
print(jnp.shape(jnp.zeros(1)))
print(test5)
print(molar_flows)
print(max_value)
print(jnp.shape(max_value))

shape_test = PerCompoundProperty(n_Butane=jnp.array(0),n_Pentane=jnp.array(0),n_Octane=jnp.array(0))
print(shape_test)
print(jnp.shape(shape_test))