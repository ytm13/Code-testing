# Sintered vs Non-sintered, comparing 5.4micron diameter particles with Biasi
# somewhat works if you ignore the biasi

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy.special import erf

data = pd.read_csv("unsintered_data.csv", header=1)

# Paramters for RNR models 
# fluid and pipe parameters 
gas_density = 1.92  # kg/m³
kine_viscosity = 1.96e-5  # m²/s
R_list = [10e-6, 20e-6]  # particle radii (10μm, 20μm) 
temperature = 750 # K

# dynamic viscosity
dynamic_viscosity = (temperature ** 1.5) * 1.458 * 10**(-6) / (temperature + 110.4) 

# Aerodynamic force couple 
drag_amp_fact = 100  # r/a value found

# surface energy for non sintered 
surface_energy = 0.56  # J/m²

# friction velocity range 
fric_velo_range = np.logspace(-1, 1, 200)  # 0.1m/s to 10m/s
t_eval = 0.01  # seconds

def adhesion(f_prime_a, sigma_prime, mean_f_prime): 
    adhe_front_divisor = np.sqrt(2 * np.pi) * f_prime_a * np.log(sigma_prime) 
    adhe_back = (-0.5) * ((np.log(f_prime_a / mean_f_prime) / np.log(sigma_prime)) ** 2)
    adhesion_forces = np.exp(adhe_back) / adhe_front_divisor
    return adhesion_forces

# no conditional 
def resuspension_p(fa, mean_force_couple, var_force, n_theta): 
    p_dividend = -((fa - mean_force_couple) **2) / (2 * var_force)
    p_divisor = 0.5 * (1 + erf((fa - mean_force_couple) / np.sqrt(2 * var_force)))
    p_values = n_theta * np.exp(p_dividend) / (p_divisor) 
    return p_values

# initialise empty dictionary to store the values
results1 = {}  # non-sintered
results2 = {}  # sintered

R = 5.4e-6 / 2

# Adhesive force for smooth surface
smooth_fa = 1.5 * np.pi * surface_energy * R

# biasi distribution parameters for unsintered
sigma_prime_biasi = 1.8 + 0.136 * R ** 1.4
mean_f_prime_biasi = 0.016 - 0.0023 * R ** 0.545

# mean and sigma for sintered, 750 degree C
sigma_prime_sinter = np.exp(0.3)
mean_f_prime_sinter = 0.013519

# initialise 
remain_frac_values_unsinter = []
remain_frac_values_sinter = []

# range for normalised adhesive forces 
f_prime_a_range = np.logspace(-4, 1, 2000)

for fric_velo in fric_velo_range:
    pv2 = gas_density * (kine_viscosity ** 2)
    inner_prod = R * fric_velo / kine_viscosity

    mean_force_couple = pv2 * 10.45 * (1 + (300 * (inner_prod)**(-0.31))) * (inner_prod ** 2.31)
    mean_fluc_force = 0.2 * mean_force_couple
    var_force = mean_fluc_force ** 2  # variance 

    # frequency of forcing motion 
    n_theta = 0.00658 * (fric_velo**2 / kine_viscosity)  # approx

    fa = f_prime_a_range * smooth_fa
        
    # unsintered adhesion pdf calculation 
    adhesion_unsinter = adhesion(f_prime_a_range, sigma_prime_biasi, mean_f_prime_biasi)
    adhesion_unsinter /= integrate.simpson(adhesion_unsinter, f_prime_a_range)
    p_unsinter = resuspension_p(fa, mean_force_couple, var_force, n_theta)  # find resuspension fraction p
    remain_frac_values_unsinter.append(integrate.simpson(adhesion_unsinter * np.exp(-p_unsinter * t_eval), f_prime_a_range)) # remain fraction fR(t)

    # sintered adhesion pdf calculation 
    adhesion_sinter = adhesion(f_prime_a_range, sigma_prime_sinter, mean_f_prime_sinter)
    adhesion_sinter /= integrate.simpson(adhesion_sinter, f_prime_a_range)
    p_sinter = resuspension_p(fa, mean_force_couple, var_force, n_theta)  # find resuspension fraction p
    remain_frac_values_sinter.append(integrate.simpson(adhesion_sinter * np.exp(-p_sinter * t_eval), f_prime_a_range)) # remain fraction fR(t)

results1[mean_f_prime_biasi] = remain_frac_values_unsinter
results2[mean_f_prime_sinter] = remain_frac_values_sinter

# plot graph
plt.figure(figsize=(8,5))

# unsintered data scatterplot
plt.scatter(data["friction velocity"], data["fraction remaining"], marker="o", label="Unsintered")

# unsintered plot
plt.plot(fric_velo_range, remain_frac_values_unsinter, 'o-', markersize=4, label='Unsintered particles')
# sintered plot
plt.plot(fric_velo_range, remain_frac_values_sinter, 'o-', markersize=4, label='Sintered particles')

plt.title("Graph showing effect of sintering on threshold u*")
plt.xlabel("friction velocity (m/s)")
plt.ylabel("remainded fraction")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()