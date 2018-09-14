import os
import numpy as np
from scipy.stats import gumbel_r
import cython

#compile pyx file
os.system('python setup_dike.py build_ext --inplace')

import pyximport; pyximport.install() #pyximport.install(pyimport = True)
from dike_model_c import frequency_failure, freq_fail_c




dike_par = {}
dike_par['slope'] = 5.51 / (6 + 6.75 + 5.77)
dike_par['crown_height'] = 7.5
dike_par['gamma_b'] = 1 # no berm
dike_par['gamma_beta'] = 1
dike_par['gamma_f'] = 0.90
dike_par['q_critical'] = 0.1



# water level in the lake
water_level=np.ones(50)

# wind model: GEV type Gumbel
wind_model = gumbel_r
# frozen at one point
mu_model, sigma_surge = 10, 5
frozen_wind_model = wind_model.freeze(loc=mu_surge,scale=sigma_surge)  # define model

water


% test dike_failure mu_model

F = frequency_failure( water_level , frozen_wind_model, dike_par)
print(F)


