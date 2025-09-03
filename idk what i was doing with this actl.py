# RNR Model trial 3 fixed resuspension p
import numpy as np

# Graphite dust Rock n Roll resuspension model 

# fluid and particle parameters 
gas_density = 1.92
fluid_velocity = 22
pipe_diameter = 0.200
temperature =  700 + 273.15
R = 10e-6  #radius of graphite particle (m)
#a =  0.4micrometres? #d btwn peaks of wall surface roughness 

# dynamic viscosity
dynamic_viscosity = (temperature ** 1.5) * 1.458 * 10**(-6) / (temperature + 110.4) 
# ^^^Hilsenrath rs that Hall used in his paper where everyone gets the experimental results from

# Reynolds Number 
Re = gas_density * fluid_velocity * pipe_diameter / dynamic_viscosity

# Friction factor 
ff = 0.158 * Re ** (-0.25)

# Friction velocity 
friction_velocity = fluid_velocity * np.sqrt(ff/2) 

# I 
turbulence_intensity = 0.16 * np.sqrt(friction_velocity/fluid_velocity)

turbulent_ke = 1.5 * (turbulence_intensity * fluid_velocity)** 2

print(friction_velocity, turbulent_ke)