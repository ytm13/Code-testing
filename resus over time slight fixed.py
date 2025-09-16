# resuspension of graphite dust over time, with biasi approximations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.integrate as integrate

# fluid and pipe parameters 
gas_density = 1.92  # kg/m³
kine_viscosity = 1.96e-5  # m²/s
temperature = 748 # 700 + 273.15 K

# particle radius
R = 5e-6 / 2

# dynamic viscosity
dynamic_viscosity = (temperature ** 1.5) * 1.458 * 10**(-6) / (temperature + 110.4) 

# Aerodynamic force couple 
drag_amp_fact = 100  # r/a value found
'''kine_viscosity = 1.96e-5  # kinematic viscosity'''
kine_viscosity = dynamic_viscosity / gas_density

# surface energies and adhesive forces 
surface_energy = 0.15  # J/m²
smooth_fa = 1.5 * np.pi * surface_energy * R  # Adhesive force for smooth surface

# biasi distribution parameters
sigma_prime_biasi = 1.8 + 0.136 * R ** 1.4
mean_f_prime_biasi = 0.016 - 0.0023 * R ** 0.545

# friction velocities to be tested 
ustar_list = [0.5, 1.0, 1.5, 2.0]

# time range x axis
time_range = np.logspace(-6, 2, 300)  # like e-6 time to 100 seconds

def adhesion(f_prime_a, sigma_prime, mean_f_prime_adhe): 
    adhe_front_divisor = np.sqrt(2 * np.pi) * f_prime_a * np.log(sigma_prime) 
    adhe_back = (-0.5) * (np.log(f_prime_a / mean_f_prime_adhe) / np.log(sigma_prime)) ** 2
    adhesion_forces = np.exp(adhe_back) / adhe_front_divisor
    return adhesion_forces

# no conditional 
def resuspension_p(fa, mean_force_couple, var_force, n_theta): 
    p_dividend = -((fa - mean_force_couple) **2) / (2 * var_force)
    p_divisor = 0.5 * (1 + erf((fa - mean_force_couple) / np.sqrt(2 * var_force)))
    p_values = n_theta * np.exp(p_dividend) / (p_divisor) 
    return p_values

# initialise empty dictionary to store the values  
result_unsinter = {}

# range for normalised adhesive forces 
f_prime_a_range = np.logspace(-6, 3, 2000)

for fric_velo in ustar_list: 
    resus_frac_unsinter = []

    # frequency of forcing motion 
    n_theta = 0.00658 * (fric_velo**2 / kine_viscosity)  # approx

    pv2 = gas_density * (kine_viscosity ** 2)
    inner_prod = R * fric_velo / kine_viscosity

    mean_force_couple = pv2 * 10.45 * (1 + 300 * (inner_prod)**(-0.31)) * inner_prod ** 2.31
    mean_fluc_force = 0.2 * mean_force_couple
    var_force = mean_fluc_force ** 2  # variance 

    for t_eval in time_range:

        # find resuspension fraction p
        fa = f_prime_a_range * smooth_fa
        p_values = resuspension_p(fa, mean_force_couple, var_force, n_theta)

        # UNSINTERED 
        # adhesion pdf calculation 
        adhesion_unsinter = adhesion(f_prime_a_range, sigma_prime_biasi, mean_f_prime_biasi)
        adhesion_unsinter /= integrate.simpson(adhesion_unsinter, f_prime_a_range)

        # resuspension fraction 
        integrand_remain_unsinter = adhesion_unsinter * np.exp(-p_values * t_eval)
        remain_frac_unsinter = integrate.simpson(integrand_remain_unsinter, f_prime_a_range)
        resus_frac_unsinter.append(1-remain_frac_unsinter)

    result_unsinter[fric_velo] = resus_frac_unsinter

# plot remain fraction against friction velocity 
plt.figure(figsize=(10, 6))
for fric_velo in ustar_list:
    plt.plot(time_range, result_unsinter[fric_velo], '-', label=f'Unsintered, u*={fric_velo} m/s')

plt.xscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Resuspension Fraction')
plt.title('Effect of standard deviation of Adhesion on Particle Resuspension\n(Rock\'n\'Roll Model,)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()