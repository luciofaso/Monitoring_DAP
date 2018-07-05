import os
import numpy as np
from scipy.stats import gumbel_r
import cython

#compile pyx file
os.system('python setup_dike.py build_ext --inplace')

import pyximport; pyximport.install() #pyximport.install(pyimport = True)
from dike_model_c import frequency_failure, freq_fail_c






slope= 5.51 / (6 + 6.75 + 5.77)
crown_height = 7.5

#gamma = Gamma
gamma_b = 1 # no berm
#gamma.xi
gamma_beta = 1
gamma_f = 0.90

q_critical=0.1


# surge model: GEV type Gumbel
model_surge = gumbel_r


average_water_level=0 # TODO For cycle

F=np.array((50,50))
mu_surge, sigma_surge = 2.1, 0.6
surge_frequency= model_surge.freeze(loc=mu_surge,scale=sigma_surge)  # define model

F = frequency_failure( average_water_level, surge_frequency, crown_height, slope, gamma_b,gamma_beta, gamma_f, q_critical)
print(F)