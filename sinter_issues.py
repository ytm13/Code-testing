# help.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy.special import erf

data = pd.read_csv("11microndataset.csv", header=1)

# Parameters for RNR models 
gas_density = 1.92       # kg/m³
kine_viscosity = 1.96e-5 # m²/s
R = 5.4e-6 /2           # particle radius (5.4 µm diameter)
temperature = 750        # K

# dynamic viscosity (Sutherland’s law)
dynamic_viscosity = (temperature ** 1.5) * 1.458e-6 / (temperature + 110.4)

# surface energy for smooth (non-sintered)
surface_energy = 0.15  # J/m²

# friction velocity range
fric_velo_range = np.logspace(-2, 1, 200)  # 0.1 m/s → 10 m/s
t_eval = 1  # seconds

# Adhesion PDF (log-normal)
def adhesion(f_prime_a, sigma_prime, mean_f_prime): 
    adhe_front = np.sqrt(2 * np.pi) * f_prime_a * np.log(sigma_prime) 
    adhe_back = -0.5 * ((np.log(f_prime_a / mean_f_prime) / np.log(sigma_prime)) ** 2)
    return np.exp(adhe_back) / adhe_front

# Resuspension probability function
def resuspension_p(fa, mean_force_couple, var_force, n_theta): 
    p_dividend = -((fa - mean_force_couple) ** 2) / (2 * var_force)
    p_divisor = 0.5 * (1 + erf((fa - mean_force_couple) / np.sqrt(2 * var_force)))
    return n_theta * np.exp(p_dividend) / p_divisor

# EXPERIMENTAL VALUES ONLY

# Smooth adhesive force baseline
smooth_fa = 1.5 * np.pi * surface_energy * R  

# Experimental means (from paper)
mean_f_prime_unsinter = 0.006   # unsintered, 5.4 µm, 750 °C
mean_f_prime_sinter   = 0.013519   # sintered, 5.4 µm, 750 °C
sigma_prime = np.exp(0.3)  # assume same sd

f_prime_a_range = np.logspace(-4, 1, 2000)

remain_frac_values_unsinter_exp = []
remain_frac_values_sinter_exp   = []

for fric_velo in fric_velo_range:
    # aerodynamic forcing
    pv2 = gas_density * (kine_viscosity ** 2)
    inner_prod = R * fric_velo / kine_viscosity
    mean_force_couple = pv2 * 10.45 * (1 + (300 * (inner_prod)**(-0.31))) * (inner_prod ** 2.31)
    mean_fluc_force = 0.2 * mean_force_couple
    var_force = mean_fluc_force ** 2  
    n_theta = 0.00658 * (fric_velo**2 / kine_viscosity)

    # Adhesion PDFs forced to experimental means
    adhesion_unsinter = adhesion(f_prime_a_range, sigma_prime, mean_f_prime_unsinter)
    adhesion_unsinter /= integrate.simpson(adhesion_unsinter, f_prime_a_range)

    adhesion_sinter = adhesion(f_prime_a_range, sigma_prime, mean_f_prime_sinter)
    adhesion_sinter /= integrate.simpson(adhesion_sinter, f_prime_a_range)

    fa_uns = f_prime_a_range * smooth_fa
    fa_sin = f_prime_a_range * smooth_fa

    p_unsinter = resuspension_p(fa_uns, mean_force_couple, var_force, n_theta)
    p_sinter   = resuspension_p(fa_sin, mean_force_couple, var_force, n_theta)

    remain_frac_values_unsinter_exp.append(integrate.simpson(adhesion_unsinter * np.exp(-p_unsinter * t_eval), f_prime_a_range))
    remain_frac_values_sinter_exp.append(integrate.simpson(adhesion_sinter * np.exp(-p_sinter * t_eval), f_prime_a_range))

# THEORETICAL WITH BIASI 

# Biasi distribution parameters
sigma_prime_biasi = 1.8 + 0.136 * R ** 1.4
mean_f_prime_biasi = 0.016 - 0.0023 * R ** 0.545

# Scale sintered case with factor relative to experimental increase
scale_factor = mean_f_prime_sinter / mean_f_prime_unsinter  

remain_frac_values_unsinter_biasi = []
remain_frac_values_sinter_biasi   = []

for fric_velo in fric_velo_range:
    pv2 = gas_density * (kine_viscosity ** 2)
    inner_prod = R * fric_velo / kine_viscosity
    mean_force_couple = pv2 * 10.45 * (1 + (300 * (inner_prod)**(-0.31))) * (inner_prod ** 2.31)
    mean_fluc_force = 0.2 * mean_force_couple
    var_force = mean_fluc_force ** 2  
    n_theta = 0.00658 * (fric_velo**2 / kine_viscosity)

    adhesion_unsinter = adhesion(f_prime_a_range, sigma_prime_biasi, mean_f_prime_biasi)
    adhesion_unsinter /= integrate.simpson(adhesion_unsinter, f_prime_a_range)

    adhesion_sinter = adhesion(f_prime_a_range, sigma_prime_biasi * np.sqrt(scale_factor), mean_f_prime_biasi * scale_factor)
    adhesion_sinter /= integrate.simpson(adhesion_sinter, f_prime_a_range)

    fa_uns = f_prime_a_range * smooth_fa
    fa_sin = f_prime_a_range * smooth_fa * scale_factor

    p_unsinter = resuspension_p(fa_uns, mean_force_couple, var_force, n_theta)
    p_sinter   = resuspension_p(fa_sin, mean_force_couple, var_force, n_theta)

    remain_frac_values_unsinter_biasi.append(integrate.simpson(adhesion_unsinter * np.exp(-p_unsinter * t_eval), f_prime_a_range))
    remain_frac_values_sinter_biasi.append(integrate.simpson(adhesion_sinter * np.exp(-p_sinter * t_eval), f_prime_a_range))


# Plot results

plt.figure(figsize=(9,6))

# Experimental from sinter experiments (zhao sinter paper)
plt.scatter(data["XU"], data["YU"], marker="o", label="Exp. Unsintered")
'''plt.scatter(data["X5"], data["Y5"], marker="o", label="Exp. Sintered 500C 9h")
plt.scatter(data["X75"], data["Y75"], marker="o", label="Exp. Sintered 750C 9h")'''
'''
# Experimental from Hall experiments (zhang rnr paper)
plt.scatter(data["X9"], data["Y9"], marker="s", label="RUN9")
plt.scatter(data["X10"], data["Y10"], marker="s", label="RUN10")
plt.scatter(data["X15"], data["Y15"], marker="s", label="RUN15")
'''
# Experimental model
plt.plot(fric_velo_range, remain_frac_values_unsinter_exp, '-', label="Unsintered (exp mean)")
plt.plot(fric_velo_range, remain_frac_values_sinter_exp, '-', label="Sintered (exp mean)")
\
# Biasi model
plt.plot(fric_velo_range, remain_frac_values_unsinter_biasi, '--', label="Unsintered (Biasi)")
plt.plot(fric_velo_range, remain_frac_values_sinter_biasi, '--', label="Sintered (Biasi-scaled)")

plt.title("Experimental vs Biasi Models")
plt.xlabel("Friction velocity u* (m/s)")
plt.ylabel("Remaining fraction")
plt.xscale("log")
plt.ylim(0,1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
