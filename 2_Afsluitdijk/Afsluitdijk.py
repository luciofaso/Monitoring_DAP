import os
from scipy.stats import gumbel_r
import cython
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


os.system('python ../model/dike/setup_dike.py build_ext --inplace')
import pyximport; pyximport.install() #pyximport.install(pyimport = True)
from models.dike.dike_model_c import frequency_failure


# averarge water level
wl_2017=0.1  # above NAP
sea_rise_year= 3.26 / 1000 # m/year

#storm characteristics
# from historical data at Den Helder (see jupiter notebook)
#
mu_surge_data = 1.90
sigma_surge_data = 0.394

# surge model: GEV type Gumbel
model_surge = gumbel_r
surge_frequency = model_surge.freeze(loc=mu_surge_data,scale=sigma_surge_data)  # specify model


wl_2050 = wl_2017 + sea_rise_year * 30


dike_par = {'height': 7.85, # dike_heights,
            'slope': 5.51 / (6 + 6.75 + 5.77), # slope,
            'gamma_b':1 ,
            'gamma_beta': 1,
            'gamma_f': 0.9,
            'q_critical':0.1
            }



dike_2050_failure = lambda average_water_level, surge_frequency:  frequency_failure( surge_frequency, dike_par , average_water_level, base_year=2000)

F_standard = np.zeros(30)
for i in range(30):
    F_standard[i] = dike_2050_failure(wl_2050, surge_frequency)
    print(F_standard)
error_model = np.std(F_standard)
print("standard deviation moder error, in yrs: " + str(error_model))
print("standard deviation moder error, in %: " + str(error_model/np.mean(F_standard)))



# estimate frequency of failure for mesh of water level and surge frequency

#HP surge frequency changes in mu only, or correlated as in literature [see literature]
range_wl_rate=np.linspace(-5,20,25)/1000   # mm/year
range_mu_surge=np.linspace(1.7,2.5,25) # mm/year 1000-year surge
#range_sigma=range()


F_df=pd.DataFrame(index=range_wl_rate*1000,columns=range_mu_surge)
F_df=F_df.rename_axis('sea level rise rate')

for wl_rise_rate in range_wl_rate:
    for mu_surge in range_mu_surge:
        wl_2050 = wl_2017 + wl_rise_rate * 30
        surge_freq = model_surge.freeze(loc=mu_surge,scale=sigma_surge_data)
        F_df.at[wl_rise_rate*1000,mu_surge]=dike_2050_failure(wl_2050,surge_freq)

print(F_df)
F_df.to_csv('data/Stress_test')

