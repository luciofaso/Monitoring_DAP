# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'


import numpy as np
from scipy.interpolate import UnivariateSpline

def sim_H(h_init, forcings, par, policy):
    """ Simulation of Lake dynamic"""


    # forcings
    q_ij = forcings['discharge Ijssel']
    q_lat = forcings['discharge lateral']
    q_dem = forcings['demamd']
    h_wz = forcings['level Wadden Sea']
    wind_v = forcings['wind velocity']
    wind_d = forcings['wind direction']

    # test that input all same lenght, otherwise throw an error
    if len(q_ij) == len(q_lat) == len(q_dem) == len(h_wz) == len (wind_v) == len(wind_d):
        H = len(q_ij)
    else:
        raise Exception('Input signals are not of the same length')


    # policy options
    h_Target_t = policy['h target']
    pump_capacity = policy['pumping capacity']

    # lake parameters
    Delta_t = par['Delta_t']
    K = par['K']
    A = par['lake surface']

    # wind parameters
    a = par['a']
    b = par['b']
    c = UnivariateSpline(range(0,391,30),par['c'])
    h_0_wind = par['h_0']


    # Create empty arrays
    h_bar = np.empty(H)
    h_wind = np.empty(H)
    q_out = np.empty(H)
    q_supply = np.empty(H)
    # u_pump = np.empty(H)

    # initialize
    h_bar[0] = h_init

    # cycle
    for t in range(1,H-1):
        h_wind[t] = wind_setup (wind_v[t], wind_d[t], h_bar[t-1], a, b, c, h_0_wind)
        h_bar[t],q_out[t],q_supply[t] = lake_sim (h_bar[t-1], h_wind[t],
                                                  q_ij[t], q_lat[t], q_dem[t],h_wz[t],
                                                  h_Target_t, pump_capacity,
                                                  Delta_t, K, A)

    return (h_bar,h_wind,q_out,q_supply)

def wind_setup (wind_v, wind_d, h_bar, a, b, c, h_0):
    """ Calculate wind set-up at n(=2) locations """

    h_wind = (c(wind_d) * wind_v ** a ) / ((h_0 + h_bar) ** b)

    return h_wind




def lake_sim(h_bar_tmin1, h_wind_t, q_ij_t, q_lat_t, q_dem_t, h_wz_t, h_Target_t, pump_capacity, Delta_t, K, A):
    """Dynamic model of the lake"""
    
    # TODO CYTHONIZE THIS FUNCTION

    q_free = K * np.sqrt(2 * 9.8 * max(h_bar_tmin1 + h_wind_t - h_wz_t, 0))
    u_release = 1 if h_bar_tmin1 >= h_Target_t else 0
    q_out = (q_free + pump_capacity) * u_release
    q_supply = q_dem_t # TODO Integrate supply function

    h_bar_t = h_bar_tmin1 + Delta_t / A * (q_ij_t + q_lat_t - q_out - q_supply)

    return h_bar_t, q_out, q_supply


