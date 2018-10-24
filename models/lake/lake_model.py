# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'


import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd

def sim_H(h_init, forcings, par_lake, par_wind, policy):
    """ Simulation of Lake dynamics"""


    # preprocess input
    #forcings['wind velocity'][forcings['wind velocity']<0] = 0 # set negative wind velocity to zero


    daily_variables = forcings.columns.drop('sea level')

    forcings_daily = forcings[daily_variables.values].resample('D').mean()


    # forcings
    q_ij = forcings_daily['inflow ijssel'].values
    q_lat = forcings_daily['inflow lateral'].values
    wind_v_mean = forcings_daily['wind speed, average'].values
    wind_d = forcings_daily['wind direction'].values
    pot_evaporation = forcings_daily['potential evaporation'].values
    q_demand = forcings_daily['water demand'].values

    n_days = len(q_ij)
    h_sea = forcings['sea level'].values[:n_days*24].reshape((n_days,24))
    # TODO: test that input arrays have no nan cells



    # policy options
    h_target = set_target_level(policy['winter target'], policy['summer target'], forcings_daily.index).values

    pump_capacity = policy['pump capacity']
    pump_head = 2 # fixed

    # lake parameters
    Delta_t = par_lake['Delta_t']
    K = par_lake['K']
    A = par_lake['Surface']
    h_supply_th = par_lake['supply threshold']

    # wind parameters
    a = par_wind['a']
    b = par_wind['b']
    c = UnivariateSpline(range(0,391,30),par_wind['c'])
    h_0_wind = par_wind['h_0']

    # Create empty arrays
    H = len(q_ij)
    h_bar = np.empty(H)
    h_wind = np.empty(H)
    q_supply = np.empty(H)
    q_free = np.empty(H)
    q_pump = np.empty(H)

    # initialize
    h_bar[0] = h_init

    # cycle
    for t in range(1,H):
        h_wind[t] = wind_setup (wind_v_mean[t], wind_d[t], h_bar[t-1], a, b, c, h_0_wind)
        h_bar[t],q_free[t], q_pump[t], q_supply[t] = lake_sim_step (h_bar[t-1], h_wind[t],
                                                q_ij[t], q_lat[t], q_demand[t],
                                                pot_evaporation[t],
                                                h_sea[t][:],
                                                h_target[t],
                                                h_supply_th,
                                                pump_head, pump_capacity,
                                                Delta_t/A, K)


    output = pd.DataFrame(np.array([h_bar, h_wind, q_free, q_pump, q_supply]).transpose(),
                          index = forcings_daily.index, columns=['average water level',
                                                                 'wind setup',
                                                                 'sluices release',
                                                                 'pump release',
                                                                 'water supply'])

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

    h_wind = (c(wind_direction) * wind_velocity ** a ) / ((h_0 - 0.4 ) ** b)

    return h_wind


def set_target_level(winter_target, summer_target, time_index):
    """

    Args:
        winter_target: target water level on the lake, winter period, i.e. from October to March
        summer_target: target water level on the lake, summer period, i.e. from April to September
        time_index: time index for which the series of target level must be set

    Returns:
        target_level (pd.Series): series of target water level

    """
    april = 3
    september = 8

    target_level = pd.Series(index=time_index)
    target_level[:] = winter_target
    target_level[np.logical_and(target_level.index.month >= april , target_level.index.month <= september)] = summer_target

    return target_level


try:
    from lake.lake_model_c import lake_sim_step_c
    print('lake model cython function successfully imported')
    lake_sim_step = lake_sim_step_c
except:
    print('lake model cython function not imported')
    def lake_sim_step(h_bar_tmin1, h_wind_t,
                      q_ij_t, q_lat_t, q_demand_t,
                      pot_evaporation_t,
                      h_sea_hourly: np.array,
                      h_target_t, h_supply_th,
                      pump_head, pump_capacity,
                      Delta_t_S, K):
        """Dynamic model of the lake

        Args:
            h_bar_tmin1 (float):
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


        Returns:
            h_bar_t, q_free, q_pump, q_demand_t, q_supply_t
        """

        # uncontrolled in and outflows
        Delta_h_bar_uncontrolled = Delta_t_S * (q_ij_t + q_lat_t ) - pot_evaporation_t
        h_bar_t = h_bar_tmin1 + Delta_h_bar_uncontrolled

        # supply
        q_supply_t = np.minimum( (h_bar_t - h_supply_th) / Delta_t_S , q_demand_t ) if h_bar_t >= h_supply_th else 0
        h_bar_t = h_bar_t - Delta_t_S * q_supply_t

        # physical max release Afsluitdijk, sluices and pumps
        q_free_max, q_pump_max = discharge_afsluitdijk( h_sea_hourly - h_bar_tmin1 - h_wind_t , K, pump_head, pump_capacity)

        # spui als het kan
        q_free = np.minimum(  (h_bar_t - h_target_t) / Delta_t_S, q_free_max) if h_bar_t >= h_target_t else 0
        h_bar_t = h_bar_t - Delta_t_S * q_free
        #pump als het moet
        q_pump = np.minimum(  (h_bar_t - h_target_t) / Delta_t_S, q_pump_max) if h_bar_t >= h_target_t else 0
        h_bar_t = h_bar_t - Delta_t_S * q_pump

        return h_bar_t, q_free, q_pump, q_supply_t



    def discharge_afsluitdijk(Delta_h:np.array, K:float, pump_head:float, nominal_pump_capacity:float):
        """Return the maximum free discharge and pumping discharge for given water levels

        Args:
            Delta_h (np.array): Difference between
                * hourly sea water level, in mNAP
                * daily water level in the lake, i.e. average water level + wind setup at sluices, in mNAP

            K (float): sluices carachteristics, see (Talsma), in m^2
            E (float): pumping power, in [XX]

        Returns:
            q_free : max free discharge through the sluices, in m^3/s
            q_pump_max: max pump, in m^3/s
        """

        q_free_max = np.mean( K * np.sqrt(2 * 9.8 * np.maximum(-Delta_h,0) ) )
        q_pump_max = np.mean( nominal_pump_capacity * np.sqrt( np.maximum( pump_head - Delta_h,0) ) )

        return q_free_max, q_pump_max



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

    h_df['h_wind'] = wind_setup(forcings['wind speed, hourly max'].values, forcings['wind direction'].values, h_bar_mean[0], wind_par['a'], wind_par['b'], c, wind_par['h_0'])
    h_df['h_t'] = h_df['h_bar'] + h_df['h_wind']

    h_year_max = h_df['h_t'].groupby(h_df.index.year).max()

    return h_year_max



