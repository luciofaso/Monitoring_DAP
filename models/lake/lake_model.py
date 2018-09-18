# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'


import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.special import ellipeinc
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

    daily_variables = forcings.columns.drop('sea level')

    forcings_daily = forcings[daily_variables].resample('D')



    # forcings
    q_ij = forcings_daily['discharge Ijssel'].values
    q_lat = forcings_daily['discharge lateral'].values
    temperature = forcings_daily['temperature'].values

    wind_v_mean = forcings['wind velocity, average'].values
    wind_v_max = forcings['wind velocity, max'].values
    wind_d = forcings['wind direction'].values

    n_days = len(q_ij)
    h_sea = forcings['sea level, max'].values[:n_days*24].reshape((n_days,24))
    # TODO: test that input arrays have no nan cells

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
    q_free = np.empty(H)
    q_pump = np.empty(H)
    q_demand = np.empty(H)
    q_supply = np.empty(H)
    u_pump = np.empty(H)

    # initialize
    h_bar[0] = h_init

    # cycle
    for t in range(1,H-1):
        h_wind[t] = wind_setup (wind_v_mean[t], wind_d[t], h_bar[t-1], a, b, c, h_0_wind)
        h_bar[t],q_free[t], q_pump[t],q_demand[t], q_supply[t] = lake_sim_step (h_bar[t-1], h_wind[t],
                                                  q_ij[t], q_lat[t], temperature[t],
                                                  h_sea[t][:],
                                                  h_target_t, pump_capacity,
                                                  Delta_t/A, K, k_evap, k_dem)


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


def lake_sim_step(h_bar_tmin1, h_wind_t,
                  q_ij_t, q_lat_t, temperature_t,
                  h_sea_t: np.array,
                  h_target_t, pump_capacity,
                  Delta_t_S, K, k_evap, k_dem
                  ):
    """Dynamic model of the lake

    Args:
        h_bar_tmin1:
        h_wind_t:
        q_ij_t:
        q_lat_t:
        temperature_t:
        delta_h_sea_t:
        shift_h_sea:
        h_target_t:
        pump_capacity:
        Delta_t_S:
        K:
        k_evap:
        k_dem:

    Returns:
        h_bar_t, q_free, q_pump, q_demand_t, q_supply_t
    """

    
    # fluxes
    q_free, q_pump_max = discharge_afsluitdijk(h_bar_tmin1 + h_wind_t, h_sea)# np.sqrt(2 * 9.8 * max(h_bar_tmin1 + h_wind_t - h_wz_t, 0))
    q_evaporation_t = k_evap * temperature_t
    q_demand_t = k_dem * temperature_t # oversimplified function of demand, proportional to temperature
    q_supply_t = q_demand_t # heroic assumption, supply always satisfied TODO set low level lake limit

    #levels
    Delta_h_bar_no_pump = Delta_t_S * (q_ij_t + q_lat_t - q_free - q_supply_t - q_evaporation_t)
    h_bar_t = h_bar_tmin1 + Delta_h_bar_no_pump
    if h_bar_t >= h_target_t:
        q_pump = np.min( Delta_t_S * (h_bar_t - h_target_t) , q_pump_max)
        h_bar_t = h_bar_t - Delta_t_S * q_pump

    return h_bar_t, q_free, q_pump, q_demand_t, q_supply_t



def discharge_afsluitdijk(h_lake: float, h_sea: np.array, K: float, E: float):
    """

    Args:
        h_lake (float): water level in the lake, i.e. average water level + wind setup at sluices, in mNAP
        h_sea (24 hours array): hourly sea water level, in mNAP
        K (float): sluices carachteristics, see (Talsma), in m^2
        E (float): pumping power, in [XX]

    Returns:
        q_free : free discharge through the sluices, in m^3/s
        q_pump_max: max pump, in m^3/s
    """

    Delta_h = h_lake - h_sea
    q_free = np.mean( K * np.sqrt(2 * 9.8 * np.max(Delta_h,0) ) )
    q_pump_max = np.mean( E /  np.max(-Delta_h,0) )

    return (q_free, q_pump_max)



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

    h_df['h_wind'] = wind_setup(forcings['wind velocity, max'].values, forcings['wind direction'].values, h_bar_mean, wind_par['a'], wind_par['b'], c, wind_par['h_0'])
    h_df['h_t'] = h_df['h_bar'] + h_df['h_wind']

    h_year_max = h_df['h_t'].groupby(h_df.index.year).max()

    return h_year_max







# TODO: TO BE DELETED

def free_discharge(Delta_h_lake, var_h_sea, Delta_t=24*3600):
    """Calculate the free discharge from the sluices

    Args:
        Delta_h_lake (float): difference between water level in the lake, sum of average lake level and wind setup at
            sluices  (in m NAP) and daily average of sea water level (in m NAP)
        var_h_sea (float): difference between the maximum and the minimum sea water level (in m)
        integral_free_discharge_AfD (function): function calculating the integral of the sqrt ( sin (t) - diff )
        Delta_t (float)=: time-step lenght (in seconds, default: one day)
    Returns:
        q_free (float): the discharge (in $m^3/s$)
    """

    pos = Delta_h_lake / (var_h_sea / 2)

    if pos >= 1:  # i.e. h_lake - shift_h_sea < delta_h_sea/2:
        t_start = np.arcsin(-pos)
        t_end = np.arcsin(pos)
        q_free = integral_free_discharge(t_start, Delta_h_lake, var_h_sea, Delta_t) \
                 - integral_free_discharge(t_end, Delta_h_lake, var_h_sea, Delta_t)
    else:
        q_free = 0

    return (q_free)



def integral_free_discharge(t, delta_h_lake, delta_h_sea, Delta_t, freq_sin=24.5*3600):
    """Calculate comulative discharge

    Args:
        t: instant of integral
        h_lake: water level in the lake, sum of average and wind setup level
        shift_h_sea:
        delta_h_sea:
        K:
        Delta_t:
        freq_sin: frequency of tide

    Returns:

    """

    a = delta_h_sea / 2
    b = delta_h_lake
    c = freq_sin / (2 * np.pi)
    k1 = np.sqrt(b-a*np.sin(c*t))
    phi = (np.pi - 2 * c * t) / 4
    m = a / ( (a-b) / 2 )

    cum_discharge = 2 * np.sqrt( 9.8 np.abs(a-b)) * k1 * ellipeinc(phi, m)
    return (cum_discharge)