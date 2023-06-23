# import csv
#
# # Open the CSV file
# with open('heat_capacity_liquid.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#
#     # Create a dictionary with the values for each component
#     component_values = {}
#     for row in reader:
#         component_values[row['Name']] = {
#             'CAS no.': row['CAS no.'],
#             'C1': float(row['C1']),
#             'C2': float(row['C2']),
#             'C3': float(row['C3']),
#             'C4': float(row['C4']),
#             'C5': float(row['C5']),
#             'Tmin [K]': float(row['Tmin [K]']),
#             'Val(Tmin)': float(row['Val(Tmin)']),
#             'Tmax [K]': float(row['Tmax [K]'])
#         }
#
# # Access the values for n-Butane, n-Pentane, and n-Octane
# n_butane_values = component_values['n-Butane']
# n_pentane_values = component_values['n-Pentane']
# n_octane_values = component_values['n-Octane']
#
# print(component_values)

# with open('heat_capacity_liquid.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     component_values = {}
#     next(reader) # skip header row
#     for row in reader:
#         name = row[0]
#         values = [float(x) for x in row[2:]]
#         component_values[name] = values
#
# print(component_values)
#
# # Access the values by component name:
# print(component_values['n-Butane'])  # Output: [191030.0, -1675.0, 12.5, -0.03874, 4.6121e-05, 134.86, 112720.0, 400.0]
# print(component_values['n-Pentane']) # Output: [159080.0, -270.5, 0.99537, 0.0, 0.0, 143.42, 140760.0, 390.0]
# print(component_values['n-Octane'])  # Output: [224830.0, -186.63, 0.95891, 0.0, 0.0, 216.38, 229340.0, 460.0]

import jax.numpy as jnp
import jax

K_func = {}
CpL_func = {}
CpV_func = {}
dH_func = {}
T_ref = {}

components = ['n-Butane', 'n-Pentane', 'n-Octane']
component_indices = {
    'n-Butane': 2,
    'n-Pentane': 3,
    'n-Octane': 4
}

with open('depriester.csv', 'r') as f:
    next(f) # skip header row
    for line in f:
        fields = line.strip().split(',')
        name = fields[0]
        if name in component_indices:
            values = jnp.array(list(map(float, fields[1:-1])))
            K_func[name] = values


with open('heat_capacity_liquid.csv', 'r') as f:
    next(f) # skip header row
    for line in f:
        fields = line.strip().split(',')
        name = fields[0]
        values = jnp.array(list(map(float, fields[2:-3])))
        CpL_func[name] = values

CpV_value = jnp.array([4.*8.314*1000])
CpV_func = {key: CpV_value for key in components}

with open('heats_of_vaporization.csv', 'r') as f:
    next(f) # skip header row
    for line in f:
        fields = line.strip().split(',')
        name = fields[0]
        dH_values = jnp.array(list(map(float, fields[2:3])))
        T_ref_values = jnp.array(list(map(float, fields[3:4])))
        dH_func[name] = dH_values
        T_ref[name] = T_ref_values

print(K_func)
print('--------------------------------------')
print(CpL_func)
print('--------------------------------------')
print(CpV_func)
print('--------------------------------------')
print(dH_func)
print('--------------------------------------')
print(T_ref)
print('--------------------------------------')

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
num_constants = 5

def eval_CpL(T, c):
    constants = CpL_func[c]
    power_law = [constants[i]*jax.lax.pow(T, (i-1.0)) for i in range(1, num_constants + 1)]
    power_law_array = jnp.array(power_law)
    result = jnp.sum(power_law_array)
    return result


print(eval_CpL(300.0,'n-Butane'))

def integral_CpL(T, c):
    constants = CpL_func[c]
    power_law = [constants[i] * jax.lax.pow(T, (i - 0.0))/i for i in range(1, num_constants + 1)]
    power_law_array = jnp.array(power_law)
    result = jnp.sum(power_law_array)
    return result

def integral_dT_CpL(T_ref, T, c):
    int_T = integral_CpL(T, c)
    int_T_ref = integral_CpL(T_ref, c)
    result = int_T - int_T_ref
    return result

print(integral_dT_CpL(300.0, 330.0, 'n-Butane'))

def integral_dT_CpV(T_ref, T, c):
    value = CpV_func[c]
    result = value * (T - T_ref)
    return result

print(integral_dT_CpV(300.0, 330.0, 'n-Butane'))
def eval_dH_func(c):
    value = dH_func[c]
    result = value * 1e6
    return result

print(eval_dH_func('n-Butane'))