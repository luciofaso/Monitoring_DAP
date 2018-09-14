# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'


import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd

import os
import cython

#compile pyx file
os.system('python model/lake/setup_lake.py build_ext --inplace')

import pyximport; pyximport.install() #pyximport.install(pyimport = True)
# from model.lake.lake_model_c import lake_sim_step_c


def sim_H(h_init, forcings, par_lake, par_wind, policy):
    """ Simulation of Lake dynamics"""


    # preprocess input
    forcings['wind velocity'][forcings['wind velocity']<0] = 0 # set negative wind velocity to zero

    # forcings
    q_ij = forcings['discharge Ijssel'].values
    q_lat = forcings['discharge lateral'].values
    temperature = forcings['temperature'].values
    h_s = forcings['sea level'].values
    wind_v = forcings['wind velocity'].values
    wind_d = forcings['wind direction'].values

    # policy options
    h_target_t = policy['target level']
    pump_capacity = policy['pumping capacity']

    # lake parameters
    Delta_t = par_lake['Delta_t']
    K = par_lake['K']
    A = par_lake['Surface']
    k_evap = par_lake['k_evap']
    k_dem = par_lake['k_dem']

    # wind parameters
    a = par_wind['a']
    b = par_wind['b']
    c = UnivariateSpline(range(0,391,30),par_wind['c'])
    h_0_wind = par_wind['h_0']

    # Create empty arrays
    H = len(q_ij)
    h_bar = np.empty(H)
    h_wind = np.empty(H)
    q_out = np.empty(H)
    q_demand = np.empty(H)
    q_supply = np.empty(H)
    u_pump = np.empty(H)

    # initialize
    h_bar[0] = h_init

    # cycle
    for t in range(1,H-1):
        h_wind[t] = wind_setup (wind_v[t], wind_d[t], h_bar[t-1], a, b, c, h_0_wind)
        h_bar[t],q_out[t],q_demand[t], q_supply[t] = lake_sim_step (h_bar[t-1], h_wind[t],
                                                  q_ij[t], q_lat[t], temperature[t],h_s[t],
                                                  h_target_t, pump_capacity,
                                                  Delta_t, K, A, k_evap, k_dem)


    output = pd.DataFrame(np.array([h_bar, h_wind, q_out, q_demand, q_supply]).transpose(),
                          index = forcings.index, columns=['h_bar', 'h_wind', 'q_out', 'q_demand', 'q_supply'])

    return (output)


def wind_setup (wind_velocity, wind_direction, h_bar, a, b, c, h_0):
    """ Calculate wind set-up at the locations of interest

    Args:
        wind_velocity: velocity of the wind, expressed in xxx ???
        wind_direction:
        h_bar:
        a, b, c: wind-setup model parameters
        h_0:

    Returns:
        h_wind (float): wind-setup, i.e. water level increase due to the wind
    """

    h_wind = (c(wind_direction) * wind_velocity ** a ) / ((h_0 + h_bar) ** b)

    return h_wind


def lake_sim_step(h_bar_tmin1, h_wind_t, q_ij_t, q_lat_t, temperature_t, h_wz_t,
                  h_target_t, pump_capacity,
                  Delta_t, K, A, k_evap, k_dem
                  ):
    """Dynamic model of the lake"""
    
    # TODO CYTHONIZE THIS FUNCTION

    q_free = K * np.sqrt(2 * 9.8 * max(h_bar_tmin1 + h_wind_t - h_wz_t, 0))
    u_release = 1 if h_bar_tmin1 >= h_target_t else 0
    q_out = (q_free + pump_capacity) * u_release
    q_evaporation_t = k_evap * temperature_t
    q_demand_t = k_dem * temperature_t # oversimplified function of demand, proportional to temperature
    q_supply_t = q_demand_t # heroic assumption, supply always satisfied TODO set low level lake limit

    h_bar_t = h_bar_tmin1 + Delta_t / A * (q_ij_t + q_lat_t - q_out - q_supply_t - q_evaporation_t)

    return h_bar_t, q_out, q_demand_t, q_supply_t





def yearly_max_wl(h_bar, forcings, wind_par):
    """Calculate the yearly max water level at the location of interest

    Args:
        h_bar (pd.Series): time series of average lake water level
        forcings (pd.DataFrame): Dataframe of forcings
            (only wind velocity and wind direction are required)
        wind_par: parameters of the relationship wind - water level at the location of interest

    Returns:
        h_year_max (pd.Series): yearly maximum water level at the location of interest

    """

    h_df = pd.DataFrame(h_bar)
    h_df.columns = ['h_bar']
    c = UnivariateSpline(range(0, 391, 30), wind_par['c'])
    h_bar_mean = h_df.mean(axis=0).values
    #wind = forcings[['wind velocity','wind direction']]

    h_df['h_wind'] = wind_setup(forcings['wind velocity'].values, forcings['wind direction'], h_bar_mean, wind_par['a'], wind_par['b'], c, wind_par['h_0'])
    h_df['h_t'] = h_df['h_bar'] + h_df['h_wind']

    h_year_max = h_df['h_t'].groupby(h_df.index.year).max()

    return h_year_max