# plotting 10μm alumina particles with RNR with Biasi and Base RNR model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf

# retrieve information from csv to plot RUNS
data1 = pd.read_csv("rnr 10micron alumina.csv", header=1)
data2 = pd.read_csv("rnr 20micron alumina.csv", header=1)

# Paramters for RNR models 
# fluid and pipe parameters 
gas_density = 1.92  # kg/m³
kine_viscosity = 1.96e-5  # m²/s
R_list = [10e-6, 20e-6]  # particle radii (10μm, 20μm) 
temperature = 700 + 273.15 # K

# dynamic viscosity
dynamic_viscosity = (temperature ** 1.5) * 1.458 * 10**(-6) / (temperature + 110.4) 

# Aerodynamic force couple 
drag_amp_fact = 100  # r/a value found
kine_viscosity = 1.96e-5  # kinematic viscosity

# surface energies and adhesive forces 
surface_energy = 0.56  # J/m²

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
results1 = {}  # biasi
results2 = {}  # non-biasi

# range for normalised adhesive forces 
f_prime_a_range = np.logspace(-8, 3, 2000)

for R in R_list: 
    # initialise empty lists 
    remain_frac_values_biasi = []
    remain_frac_values_nonbiasi = []

    # Adhesive force for smooth surface
    smooth_fa = 1.5 * np.pi * surface_energy * R  

    # biasi distribution parameters for differing particle size
    sigma_prime_biasi = 1.8 + 0.136 * R ** 1.4
    mean_f_prime_biasi = 0.016 - 0.0023 * R ** 0.545

    # non-biasi base distribution parameters for differing particle size
    mean_f_prime_nonbiasi = 0.027
    sigma_prime_nonbiasi = 10.4

    for fric_velo in fric_velo_range:
        pv2 = gas_density * (kine_viscosity ** 2)
        inner_prod = R * fric_velo / kine_viscosity

        mean_force_couple = pv2 * 10.45 * (1 + 300 * (inner_prod)**(-0.31)) * inner_prod ** 2.31
        mean_fluc_force = 0.2 * mean_force_couple
        var_force = mean_fluc_force ** 2  # variance 

        # frequency of forcing motion 
        n_theta = 0.00658 * (fric_velo**2 / kine_viscosity)  # approx

        fa = f_prime_a_range * smooth_fa
            
        # biasi adhesion pdf calculation 
        adhesion_biasi = adhesion(f_prime_a_range, sigma_prime_biasi, mean_f_prime_biasi)
        adhesion_biasi /= np.trapezoid(adhesion_biasi, f_prime_a_range)
        p_biasi = resuspension_p(fa, mean_force_couple, var_force, n_theta)  # find resuspension fraction p
        remain_frac_values_biasi.append(np.trapezoid(adhesion_biasi * np.exp(-p_biasi * t_eval), f_prime_a_range)) # remain fraction fR(t)

        # base non-biasi adhesion pdf calculation 
        adhesion_nonbiasi = adhesion(f_prime_a_range, sigma_prime_nonbiasi, mean_f_prime_nonbiasi)
        adhesion_nonbiasi /= np.trapezoid(adhesion_nonbiasi, f_prime_a_range)
        p_nonbiasi = resuspension_p(fa, mean_force_couple, var_force, n_theta)  # find resuspension fraction p
        remain_frac_values_nonbiasi.append(np.trapezoid(adhesion_nonbiasi * np.exp(-p_nonbiasi * t_eval), f_prime_a_range)) # remain fraction fR(t)

        ''' # instantaneous resuspension rate lambda
        integrand_lambda = adhesion_values * p_values * np.exp(-p_values * t_eval)
        p_rate_lambda = np.trapezoid(integrand_lambda, f_prime_a_range)
        p_rate_lambda_values.append(p_rate_lambda) '''
    results1[mean_f_prime_biasi] = remain_frac_values_biasi
    results2[mean_f_prime_nonbiasi] = remain_frac_values_nonbiasi

    if R == 10e-6:
        plt.figure(figsize=(8,5))

        # RUN plots
        plt.scatter(data1["X9"], data1["Y9"], marker="o", label="RUN9")
        plt.scatter(data1["X10"], data1["Y10"], marker="s", label="RUN10")
        plt.scatter(data1["X15"], data1["Y15"], marker="^", label="RUN15")

        # biasi plot
        plt.plot(fric_velo_range, remain_frac_values_biasi, 'o-', markersize=4, label='Biasi RNR Model')
        # non biasi plot
        plt.plot(fric_velo_range, remain_frac_values_nonbiasi, 'o-', markersize=4, label='Base RNR Model')

        plt.title("10μm alumina particles")
        plt.xlabel("friction velocity (m/s)")
        plt.ylabel("remainded fraction")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(8,5))

        plt.scatter(data2["X7"], data2["Y7"], marker="o", label="RUN7")
        plt.scatter(data2["X8"], data2["Y8"], marker="s", label="RUN8")
        plt.scatter(data2["X20"], data2["Y20"], marker="^", label="RUN20")

        # biasi plot
        plt.plot(fric_velo_range, remain_frac_values_biasi, 'o-', markersize=4, label='Biasi RNR Model')
        # non biasi plot
        plt.plot(fric_velo_range, remain_frac_values_nonbiasi, 'o-', markersize=4, label='Base RNR Model')

        plt.title("20μm alumina particles")
        plt.xlabel("friction velocity (m/s)")
        plt.ylabel("remainded fraction")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()