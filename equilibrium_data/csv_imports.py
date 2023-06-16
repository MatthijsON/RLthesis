import jax.numpy as jnp
from jumanji import ROOT_DIR, os

def get_constants(components):
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

    return K_func, CpL_func, CpV_func, dH_func, T_ref
