# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r
import sys
sys.path.append('/Users/lraso/Dropbox/Monitoring_for_DAPP/Experiments/') #TODO make this general
from models.dike.dike_model_c import frequency_failure
from models.lake.lake_model import sim_H as lake_sim
from models.lake.lake_model import yearly_max_wl
import pyximport; pyximport.install() #pyximport.install(pyimport = True)

#compile pyx file
# os.system('python dike/setup_dike.py build_ext --inplace')

# external forcings
forcings = pd.read_csv('../data/Ijsselmeer/forcings.csv',index_col=0,parse_dates=[0])
forcings = forcings['1965':'2013']

# Lake parameters
lake_par = {}
lake_par['Delta_t'] = 3600 * 24 #   daily time-step
lake_par['Surface'] = 1.182 * 10**9 # m^2
lake_par['supply threshold'] = -0.5

C = 0.88 # lateral contraction coefficient, [-];
MU = 0.66 # contraction coefficient, [-];
W_SIGMA = 120 + 180 # crest width, [m];
H_OP = 6.9 + 6.9 # opening height, [m];
lake_par['K'] = C * MU * W_SIGMA * H_OP

# dike parameters
dike_par = {}
dike_par['slope'] = 5.51 / (6 + 6.75 + 5.77)
dike_par['height'] = 6.4 #7.5
dike_par['gamma_b'] = 1 # no berm
dike_par['gamma_beta'] = 1
dike_par['gamma_f'] = 0.90
dike_par['q_critical'] = 0.1

# wind model parameters
wind_par = {}

c_location =  np.array(  # estimation of c parameter for the sluices at the Aflsuitdijk, average of the two sluices
# Stevinsluizen:
[[-0.0049,0.00033,0.00547,0.00914,0.01037,0.00882,0.0049,-0.00033,-0.00547,-0.00914,-0.01037,-0.00882,-0.0049,-0.0049],
# Lorentsluis
[-0.00715,-0.00591,-0.00308,0.00058,0.00407,0.00648,0.00715,0.00591,0.00308,-0.00058,-0.00407,-0.00648,-0.00715,-0.00715]]
)
c_AfD = np.mean(c_location,0)
c_rogge = np.array(
    [0.01346,0.00404,-0.00186,-0.00603,-0.01804,-0.01279,-0.00652,-0.00257,-0.00164,0.01153,0.0185,0.01953, 0.01346,0.01346])

wind_par['Afsluitdijk'] = dict (a = np.sqrt(1.89 * 2.05), b = 1, c = c_AfD, h_0 = 4.5)   # par a is the geometric mean of the a coefficient for Stevinsluizen and Lorentsluis
wind_par['Roggebotsluizen'] = dict (a = 2 , b = 0.6, c = c_rogge, h_0 = 4.5 )

# Initial condions
h_0 = -0.4

def lake_dike_system(h_0,forcings, parameters_cc,lake_par,dike_par,wind_par, policy):
    """Simulate the entire lake + dike system

    Args:
        h_0 (float): Initial condition, water level in the lake at t = 0
        forcings (pd.Dataframe): External forcing historically observed
        parameters_cc (dict): list of parameters that set the change in the historically observed forcings
        lake_par (dict): dictionary of parameters for the model of the lake
            K:
            A: lake surface #TODO modify
        dike_par (dict): dictionary of parameters for the model of the dike
            slope:
            crown_height:
            gamma_b:
            gamma_beta:
            gamma_f:
            q_critical:
        wind_par (dict): dictionary of parameters for the model of the wind effects on the water level

        policy (dict):
            pumping_capacity
            h_target

    Returns:
        F (float): Frequency of dike failure
    """

    # Variate Forcings (bottom-up climate change analysis)
    forcings_cc = variate_forcings(forcings,parameters_cc)

    # model of water demand
    # oversimplified model: water demand proportional to temperature
    forcings['water demand'] = 0
    k_demand = 400 / 25  # m^3/s /  T
    forcings['water demand'] = forcings.loc[:, ['temperature', 'water demand']].max(axis=1) * k_demand

    # model of rainfall-runoff
    S_lat = 1419 * 1000000  # km^2 to m^2
    alpha = 0.5
    forcings_cc['inflow lateral'] = forcings_cc['precipitation'] * S_lat * alpha / lake_par['Delta_t']  # rational formula

    # Lake simulation
    model_output = lake_sim(h_0, forcings_cc, lake_par,wind_par['Afsluitdijk'], policy)

    # Dike boundary conditions
    h_year_max = yearly_max_wl(model_output['average water level'],forcings.resample('D').mean(),wind_par['Roggebotsluizen'])
    mu_wl,sigma_wl = gumbel_r.fit(h_year_max.values)
    water_level_pdf = gumbel_r.freeze(loc=mu_wl,scale=sigma_wl) # TEST THIS

    # Dike failure
    F = frequency_failure(water_level_pdf, dike_par, base_year=100, constant_waves = True)

    return F


def variate_forcings(forcings,parameters_cc):
    """ variate the external forcings to simulate climatic change

    Args:
        forcings (pd.DataFrame):  external forcings
        parameters_cc (dict): list of dictionaries specifying, for each item :
            name, i.e. to which forcing the variation must be applied
            func, type of variation (example: np.add, or np.multiply)
            value, parameters of the func

    Returns:
        forcings_cc (pd.DataFrame): modified forcings

    """
    forcings_cc = forcings

    for parameter in parameters_cc:
        ts = forcings[parameter['name']]
        ts_cc = ts.apply(lambda row: parameter['func'](row, parameter['value']))
        forcings_cc[parameter['name']]  = ts_cc

    return forcings_cc


ijsselmeer = lambda policy,parameters_cc: lake_dike_system(h_0,forcings,parameters_cc,lake_par,dike_par,wind_par, policy)


def test_ijsselmeer_cc(summer_target = -0.2,
                       winter_target = -0.4,
            pump_capacity = 100,
            pump_power = 0,
            wind_multiplier = 1,
            sea_level_shift = 0,
            peak_inflow_multiplier = 1,
            low_flow_shift = 0,
            temperature_shift = 0):

    """ test the ijsselmeer system under different scenario of climate change, and a given policy

    Climate change conditions are simulated by variating the timeseries of the external forcings
    """
    # climate change parameters (critical uncertaintes)
    wind_cc = {'name': 'wind speed, hourly max', 'func': np.multiply, 'value': wind_multiplier}
    sea_level_cc = {'name': 'sea level', 'func': np.add, 'value': sea_level_shift}
    peak_inflow_cc = {'name': 'inflow ijssel', 'func': np.multiply, 'value': peak_inflow_multiplier}
    low_flow_cc = {'name': 'inflow ijssel', 'func': np.add, 'value': low_flow_shift}
    temperature_cc = {'name': 'temperature', 'func': np.add, 'value': temperature_shift}

    parameters_cc = [wind_cc, sea_level_cc, peak_inflow_cc, low_flow_cc, temperature_cc]

    policy = {'summer target':summer_target, 'winter target': winter_target,
              'pump capacity':pump_capacity, 'pump power': pump_power}

    F = ijsselmeer(policy,parameters_cc)
    yearly_deficit_supply = sum(forcings['water demand'].resample('D') - lake_model_output['water supply']) / \
                            ( (forcings.index[-1] - forcings.index[0]).days / 365.25 )

    return [F,yearly_deficit_supply] # it must be a list, for the EMA workbench



## test running
# F = test_ijssel_cc(ijsselmeer,policy)
# print(F)
