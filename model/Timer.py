import distillation.amundson_1958 as am
import numpy as np
import jax.numpy as jnp
import timeit


model = am.Model(
    components = [' n-Butane', ' n-Pentane', 'n-Octane'],
    F = 1000,
    P = 2*1e5,
    z_feed = [0.20, 0.5, 0.3],
    RR = 1,
    D = 400,
    N = 30,
    feed_stage = 15,
)

def timer(n_times):
    start = timeit.default_timer()
    for i in range(n_times):
        model = am.Model(
            components=[' n-Butane', ' n-Pentane', 'n-Octane'],
            F=1000,
            P=2 * 1e5,
            z_feed=[0.20, 0.5, 0.3],
            RR=1,
            D=400,
            N=30,
            feed_stage=15,
        )
    time_req = timeit.default_timer() - start
    return time_req

n_times_run = [1]
time_taken = []

n_times_test = 10

for i in range(n_times_test):
    for j in n_times_run:
        model_time = timer(j)
        time_taken.append(model_time)

print(time_taken)

