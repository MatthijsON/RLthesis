import distillation.amundson_1958 as am
import timeit
if __name__ == '__main__':

    start = timeit.default_timer()
    model = am.Model(
        components=['n-Butane', 'n-Pentane', 'n-Octane'],
        F=1000., # kmol/h
        P=2*1e5, # Pa
        z_feed = [0.20, 0.5, 0.3],
        RR=1.,
        D=400.,
        N=30,
        feed_stage=15,
    )

    time_taken = timeit.default_timer() - start
    print('TIME TAKEN IS EQUAL TO ---------------', time_taken)

    model.add_parameters(verbose=True)


    print(model.T_feed)
    model.T_feed = model.bubble_T_feed()
    print(model.T_feed)

    for i in model.stages:
        #model.T[i] = model.T_feed
        model.T = model.T.at[i].set(model.T_feed)
    print(model.T)

    print(model.L)
    print(model.V)
    model.initialize_flow_rates()
    print(model.L)
    print(model.V)

    model.update_K_values()

    for i in model.components:
        print(i, model.l[i])
    for i in model.components:
        model.solve_component_mass_bal(i)
    for i in model.components:
        print(i, model.l[i])

    print(model.T)
    for stage in model.stages:
        #model.T[stage] = model.bubble_T(stage)
        model.T = model.T.at[stage].set(model.bubble_T(stage))
    print(model.T)

    print(model.T_is_converged())

    iter = 0
    while not model.T_is_converged():
        model.update_K_values()
        for i in model.components:
            model.solve_component_mass_bal(i)
        for stage in model.stages:
            #model.T[stage] = model.bubble_T(stage)
            model.T = model.T.at[stage].set(model.bubble_T(stage))
        print(iter, model.T)
        iter += 1

    print(model.L)
    print(model.V)
    model.solve_energy_balances()
    print(model.L)
    print(model.V)

    print(model.flow_rates_converged())

    outer_loop = 0
    inner_loop = 0
    while not model.flow_rates_converged():
        outer_loop += 1
        for i in model.components:
            model.solve_component_mass_bal(i)
        for stage in model.stages:
            #model.T[stage] = model.bubble_T(stage)
            model.T = model.T.at[stage].set(model.bubble_T(stage))
        while not model.T_is_converged():
            inner_loop += 1
            model.update_K_values()
            for i in model.components:
                model.solve_component_mass_bal(i)
            for stage in model.stages:
                #model.T[stage] = model.bubble_T(stage)
                model.T = model.T.at[stage].set(model.bubble_T(stage))
        model.solve_energy_balances()
        print(outer_loop, inner_loop, model.V)

    x = {}
    for i in model.components:
        x[i] = model.l[i][:]/model.L[:]
    print(x)

    # time_taken = timeit.default_timer() - start
    # print(time_taken)

    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(model.stages, model.T, 'o')
    ax.set_xlabel('Stage Number')
    ax.set_ylabel('Temperature [K]')

    # plot liquid-phase mole fractions
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    # calculate mole fractions
    for i in model.components:
        ax2.plot(model.stages, x[i], label=i)
    ax2.set_ylabel('Liquid phase mole fraction')
    ax2.set_xlabel('Stage Number')
    ax2.legend()
    plt.show()
