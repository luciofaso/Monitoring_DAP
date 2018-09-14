import os
import numpy as np
from scipy.stats import gumbel_r
import cython

#compile pyx file
os.system('python setup_dike.py build_ext --inplace')

import pyximport; pyximport.install() #pyximport.install(pyimport = True)
from model.dike.dike_model_c import frequency_failure




dike_par = {}
dike_par['slope'] = 5.51 / (6 + 6.75 + 5.77)
dike_par['height'] = 7.8
dike_par['gamma_b'] = 1 # no berm
dike_par['gamma_beta'] = 1
dike_par['gamma_f'] = 0.90
dike_par['q_critical'] = 0.1



# water level in the lake
water_level=np.ones(50)

# wind model: GEV type Gumbel
# frozen at one point
mu_surge_data = 1.90
sigma_surge_data = 0.394

frozen_surge_model = gumbel_r.freeze(loc=mu_surge_data,scale=sigma_surge_data)  # define model

wl_2050 = 0.1 + 3.26/1000 * 30

# test dike_failure mu_model

for i in range(5):
    F = frequency_failure( frozen_surge_model , dike_par )
    print(F)




dike_2050_failure = lambda average_water_level, surge_frequency:  frequency_failure( surge_frequency, dike_par , average_water_level, base_year=2000)

for i in range(5):
    F_standard=dike_2050_failure(wl_2050,frozen_surge_model)
    print(F_standard)
