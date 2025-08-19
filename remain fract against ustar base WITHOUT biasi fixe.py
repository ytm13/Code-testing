# remain fraction against u* for 10μm alumina particle base RNR model without Biasi

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

#KEEP ALL THIS THE SAME AS BIASI
# fluid and pipe parameters 
gas_density = 1.92  # kg/m³
kine_viscosity = 1.96e-5  # m²/s
R = 10e-5  # particle radius (10μm) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
temperature = 700 + 273.15 # K

# dynamic viscosity
dynamic_viscosity = (temperature ** 1.5) * 1.458 * 10**(-6) / (temperature + 110.4) 

# Aerodynamic force couple 
drag_amp_fact = 100  # r/a value found
kine_viscosity = 1.96e-5  # kinematic viscosity

# surface energies and adhesive forces 
surface_energy = 0.56  # J/m²
smooth_fa = 1.5 * np.pi * surface_energy * R  # Adhesive force for smooth surface
#END OF KEEP SAME

# non-biasi base distribution parameters for differing particle size
mean_f_prime_nonbiasi = 0.027
sigma_prime_nonbiasi = 10.4

# friction velocity range 
fric_velo_range = np.logspace(-2, 2, 200)  # 0.1m/s to 10m/s
t_eval = 1.0  # seconds

def adhesion(f_prime_a, sigma_prime, mean_f_prime): 
    adhe_front_divisor = np.sqrt(2 * np.pi) * f_prime_a * np.log(sigma_prime) 
    adhe_back = (-0.5) * (np.log(f_prime_a / mean_f_prime) / np.log(sigma_prime)) ** 2
    adhesion_forces = np.exp(adhe_back) / adhe_front_divisor
    return adhesion_forces

# no conditional 
def resuspension_p(fa, mean_force_couple, var_force, n_theta): 
    p_dividend = -((fa - mean_force_couple) **2) / (2 * var_force)
    p_divisor = 0.5 * (1 + erf((fa - mean_force_couple) / np.sqrt(2 * var_force)))
    p_values = n_theta * np.exp(p_dividend) / (p_divisor) 
    return p_values

# initialise empty dictionary to store the values
results2 = {}

# range for normalised adhesive forces 
f_prime_a_range = np.logspace(-8, 3, 2000)

remain_frac_values_nonbiasi = []

for fric_velo in fric_velo_range:
    pv2 = gas_density * (kine_viscosity ** 2)
    inner_prod = R * fric_velo / kine_viscosity

    mean_force_couple = pv2 * 10.45 * (1 + 300 * (inner_prod)**(-0.31)) * inner_prod ** 2.31
    mean_fluc_force = 0.2 * mean_force_couple
    var_force = mean_fluc_force ** 2  # variance 

    # frequency of forcing motion 
    n_theta = 0.00658 * (fric_velo**2 / kine_viscosity)  # approx
        
    # adhesion pdf calculation 
    adhesion_values = adhesion(f_prime_a_range, sigma_prime_nonbiasi, mean_f_prime_nonbiasi)
    adhesion_values /= np.trapezoid(adhesion_values, f_prime_a_range)
    fa = f_prime_a_range * smooth_fa

    # find resuspension fraction p
    p_values = resuspension_p(fa, mean_force_couple, var_force, n_theta)

    ''' # instantaneous resuspension rate lambda
    integrand_lambda = adhesion_values * p_values * np.exp(-p_values * t_eval)
    p_rate_lambda = np.trapezoid(integrand_lambda, f_prime_a_range)
    p_rate_lambda_values.append(p_rate_lambda) '''

    # remain fraction fR(t)
    integrand_remain = adhesion_values * np.exp(-p_values * t_eval)
    remain_frac = np.trapezoid(integrand_remain, f_prime_a_range)
    remain_frac_values_nonbiasi.append(remain_frac)

results2[mean_f_prime_nonbiasi] = remain_frac_values_nonbiasi

# plot remain fraction against friction velocity 
plt.figure(figsize=(10, 6))
for mean_f_prime, remain_frac_values_nonbiasi in results2.items():
    plt.plot(fric_velo_range, remain_frac_values_nonbiasi, 'o-', markersize=4, label='Base RNR Model')

plt.xscale('log')
plt.xlabel('Friction Velocity $u_*$ [m/s]')
plt.ylabel('Remaining Fraction')
plt.title('Effect of Mean Adhesion on Particle Resuspension\n(Rock\'n\'Roll Model, t=1s)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()