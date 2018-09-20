# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'

"""
Model adapted from xx
# Full information to be found in
# Wave run-up and overtopping Jentsje W. van der Meer
"""

cimport numpy as np
import numpy as np

#from scipy.stats import gumbel_r, norm


from libc.math cimport sqrt, log, exp





def frequency_failure(surge_level_pdf, dike_par, average_water_level = 0, N=10000, base_year=300, bint constant_waves=False):
    """calculate the frequency of failure, by montecarlo using a raw implementation of importance sampling

    Args:
        water_level :
        dike_par: dictionary defining the parameters of the dike
            slope:
            crown_height:
            gamma_b:
            gamma_beta:
            gamma_f:
            q_critical:
        N: Number of samples in montecarlo method
        base_year: lowest level of return period, base for importance sampling

    Returns:
        freq_failure: frequency of dike failure
    """

    crown_height = dike_par['height']
    slope = dike_par['slope']
    gamma_b = dike_par['gamma_b']
    gamma_beta = dike_par['gamma_beta']
    gamma_f = dike_par['gamma_f']
    q_critical = dike_par['q_critical']

    surge_sample = surge_level_pdf.rvs(N * base_year)
    surge_base_year = surge_level_pdf.ppf(1 - 1 / base_year)  #NOT WORKING IN CYTHON
    surge_importance_sample = surge_sample[surge_sample >= surge_base_year]

    freq_failure=freq_fail_importance_sampling(average_water_level, surge_importance_sample ,
                                               crown_height, slope, gamma_b, gamma_beta, gamma_f, q_critical,
                                               N, base_year, constant_waves)

    return freq_failure




cdef freq_fail_importance_sampling ( float average_water_level,
                                    np.ndarray[np.double_t,ndim=1] surge_level, # hydraulic boundary conditions
                                    float crown_height,float slope, #dike parameters
                                    float gamma_b,float gamma_beta,float gamma_f,float q_critical,  #dike parameters
                                    float N,
                                    int base_year, # importance sampling parameters
                                    bint constant_waves):
    """ Calculate the frequency of failure for a sample of surge water levels, for given dike parameters
    use the importance sampling method, i.e.

     """

    cdef:
        int i, N_imp_sample
        int M = 100 # number of sampling for each evaluation of occurred discharge
        double n_failure, prob_of_failure, frequency_of_failure
        np.ndarray[np.double_t, ndim=1] q_occ


    N_imp_sample = len(surge_level)
    n_failure = 0
    for i in range(N_imp_sample):
        q_occ = occourred_discharge(average_water_level, surge_level[i], crown_height, slope, gamma_b, gamma_beta, gamma_f, M, constant_waves)
        n_failure += cond_prob_dike_failure(q_occ, q_critical)

    prob_of_failure = n_failure / ( N * base_year ) #TODO if prob_of_failure is 0, numeric problem
    #if prob_of_failure==0:
    frequency_of_failure = 1 / prob_of_failure

    return frequency_of_failure



# CYTHON PART STARTS HERE

cdef float g = 9.8  # gravity accel m^2/s


cdef tuple wave_wind_model_ijssel(float wind_velocity):
    """calculate specific wave height and specific wave period from wind velocity

    Parameters estimated from "Bottema, Marcel. Measured wind-wave climatology lake Ijsel (NL):
    main results for the period 1997-2006. 2007."

    """

    cdef:
        float m_H = 0.06666
        float m_T = 0.12666
        float T_0 = 1.34
        float wave_height, wave_period

    wave_height = m_H * wind_velocity
    wave_period = T_0 + m_T * wind_velocity


    return wave_height,wave_period


cdef tuple hydraulic_conditions(float SWL):
    """ calculation of hydraulic conditions, considering xxx asyntoptic behaviour and stationarity of relationships.

    :param SWL: surge storm water level [m]
    reference_wl : shift of reference, [m], default is 0
    :return: wave_height, wave characteristics [m], wave_period, spectral wave period [s]
    """

    cdef:
        float C1 = 2.70
        float C2 = -1.70
        float C3 = 0.1
        float wave_height
        float wave_period

    wave_height = C1 * log(SWL) + C2
    wave_period = sqrt(wave_height / C3)

    return wave_height, wave_period


cdef double cond_prob_dike_failure(np.ndarray[np.float64_t,ndim=1] q_occ, float q_critical):
    """conditional probability of dike failure

    Args:
        q_occ (array): occurring overtopping discharge
        q_critical (float): critical dike discharge
    
    Returns:
        Z (double): probability of dike failure 
    
    """

    cdef:
        int i, N
        double m_o = 1 # to add probabilistic evaluation, make m_o and m_c stochastic
        double m_c = 1
        double z, Z

    N=len(q_occ)
    z = 0
    for i in range(N):
        if m_o * q_occ[i] > m_c * q_critical:
            z += 1
    Z = z / N
    return Z


cdef np.ndarray occourred_discharge(float average_water_level,
                                    float surge_level,
                                    float crown_height,
                                    float slope,
                                    float gamma_b,
                                    float gamma_beta,
                                    float gamma_f,
                                    int N,
                                    bint constant_waves): #hydraulic conditions
    """ sample of occurred discharges, given a water lavel, surge level and dike carachteristics

    :param average_water_level: average long term water level
    :param surge_level: water level at storm, relative to average water level
    :param dike: dike caractheristics
    :param gamma: parameters, as defined in ref
    :param N: sample dimension
    :return: q_occ: occurred discharge, array of dimension N
    
    """
    cdef:
        int i
        double wave_height, wave_period, R_c, C4, C5, Q_b, Q_n
        np.ndarray[double, ndim = 1] q_occ = np.empty(N)

    if constant_waves:
        wave_height = 2.3
        wave_period = 4.8
    else:
        wave_height, wave_period = hydraulic_conditions(surge_level)

    xi_0 = relative_wave_runup(wave_height, wave_period, slope)
    water_level = average_water_level + surge_level
    R_c = crown_height - water_level  # creast free board


    for i in range(N):
        # discharge_breaking_waves
        C4 = np.random.normal(4.7, 0.55)  # or 5.2
        Q_b = 0.067 / sqrt(slope) * gamma_b * xi_0 * \
              exp(- C4 * R_c / (wave_height * xi_0 * gamma_b * gamma_f * gamma_beta))
        # discharge_non_breaking_waves
        C5 = np.random.normal(2.3, 0.35)  # or 2.6
        Q_n = 0.2 * exp(- C5 * R_c / (wave_height * gamma_f * gamma_beta))
        q_occ[i] = min(Q_b, Q_n) * sqrt(g * wave_height ** 3)  # m^3 /m / sec

    return (q_occ)


cdef float relative_wave_runup(float wave_height, float T_s, float slope):
    """calculate relative wave runup from storm and dike characteristics

    :param wave_height: Wave characteristics [m]
    :param T_s: Spectral wave period [s]
    :param slope: dike average slope [-]
    :return: xi_0, relative wave runup [m]
    """

    cdef float s_0, xi

    # s_0 is wave stepness
    s_0 = (2 * 3.14 * wave_height) / (g * T_s ** 2)
    xi_0 = slope / sqrt(s_0)

    return xi_0









