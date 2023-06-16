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