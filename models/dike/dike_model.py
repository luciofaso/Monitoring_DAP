# -*- coding: utf-8 -*-
__author__      = 'Luciano Raso'
__copyright__   = 'Copyright 2018'
__license__     = 'GNU GPL'


"""Model adapted from the van-Meer 
reference: [here](link)

"""

# Import
import numpy as np
from scipy.stats import gumbel_r, norm


global G
G = 9.8  # gravity accel m^3/s

# gamma v = reduction factor due to a vertical wall on a slope (-). NOT INCLUDED



def hydraulic_conditions(SWL):
    """ calculation of hydraulic conditions, considering xxx asyntoptic behaviour and stationarity of relationships.

    :param SWL: surge storm water level [m]
    :return: H_s, wave characteristics [m], Tm, spectral wave period [s]
    """

    C1 = 2.70
    C2 = -1.70
    C3 = 0.1

    H_s = C1 * np.log(SWL) + C2
    Tm = np.sqrt(H_s / C3)

    return H_s, Tm



def cond_prob_dike_failure(q_occ, q_critical):
    """

    :param q_occ: occurring overtopping discharge
    :param q_critical: critical dike discharge
    :return:
    """
    # to add probabilistic evaluation, make m_o and m_c stochastic
    m_o, m_c = 1, 1
    Z = sum(m_o * q_occ > m_c * q_critical) / len(q_occ)

    return Z



def frequency_failure(average_water_level, surge_frequency, dike, gamma, q_critical):
    """

    :param average_water_level:
    :param surge_frequency:
    :param dike:
    :param gamma:
    :param q_critical:
    :return:
    """

    # raw implementation of importance sampling
    base_year = 100
    N = 10000 # sampling dimension (expected) used in the importance sampling
    M = 100 # sampling dimension used at each evaluation of the importance sampling


    surge_sample = surge_frequency.rvs(N * base_year)
    surge_base_year = surge_frequency.ppf(1 - 1 / base_year)
    surge_importance_sample = surge_sample[surge_sample >= surge_base_year]
    N_imp_sample = len(surge_importance_sample)

    n_failure = 0
    for i in range(N_imp_sample):
        surge_level = surge_importance_sample[i]
        q_occ = occourred_discharge(average_water_level, surge_level, dike, gamma, M)
        n_failure += cond_prob_dike_failure(q_occ, q_critical)

    prob_of_failure = n_failure / N / base_year
    frequency_of_failure = 1 / prob_of_failure

    return frequency_of_failure






def occourred_discharge(average_water_level, surge_level, dike, gamma, N):
    """ sample of occurred discharges, given a water lavel, surge level and dike carachteristics

    :param average_water_level: average long term water level
    :param surge_level: water level at storm, relative to average water level
    :param dike: dike caractheristics
    :param gamma: parameters, as defined in ref
    :param N: sample dimension
    :return:
    """


    H_s, Tm = hydraulic_conditions(surge_level)


    gamma['xi'] = relative_wave_runup(H_s, Tm, dike['average slope'])
    # if no berm, gamma b is 1
    gamma['b'] = 1  # gamma_b(SWL, dike, z) use 1, otherwise need to iterate
    # z=wave_runup(gamma, H_s) #z not used

    R_c = dike['crown height'] - (average_water_level + surge_level)  # creast free board

    # discharge_breaking_waves
    C4 = norm(4.7, 0.55)  # or 5.2
    Q_b = 0.067 / np.sqrt(dike['average slope']) * gamma['b'] * gamma['xi'] * \
          np.exp(- C4.rvs(N) * R_c / (H_s * gamma['xi'] * gamma['b'] * gamma['f'] * gamma['beta']))

    # discharge_non_breaking_waves
    C5 = norm(2.3, 0.35)  # or 2.6
    Q_n = 0.2 * np.exp(- C5.rvs(N) * R_c / (H_s * gamma['f'] * gamma['beta']))

    q_occ = np.minimum(Q_b, Q_n) * np.sqrt(G * H_s ** 3)  # m^3 /m / sec

    return (q_occ)




def relative_wave_runup(H_s, T_s, tan_alpha):
    """calculate relative wave runup from storm and dike characteristics

    :param H_s: Wave characteristics [m]
    :param T_s: Spectral wave period [s]
    :param tan_alpha: dike average slope [-]
    :return: xi, relative wave runup [m]
    """

    # s_0 is wave stepness
    s_0 = (2 * np.pi * H_s) / (G * T_s ** 2)
    xi = tan_alpha / np.sqrt(s_0)

    return xi






# UNUSED FUNCTIONS:
def gamma_b(SWL, dike, z):
    d_h = SWL - dike['berm height']
    r_b = dike['berm width'] / dike['berm length']

    if -d_h > z or d_h > 2 * H_s:  # outside influence zone
        r_dh = 1
    else:
        if z > - d_h:  # berm above SWL
            x = z
        elif 2 * H_s > d_h:  # berm below SWL
            x = 2 * H_s

        r_dh = 0.5 - 0.5 * np.cos(np.pi * d_h / x)

    gamma_b = 1 - r_b * (1 - r_dh)

    # bounded between 0.6 and 1
    gamma_b = min(max(0.6, gamma_b), 1)
    return gamma_b


def wave_runup(gamma, H_s):
    C1 = 1.75
    C2 = 4.3
    C3 = 1.6

    # total reduction bounded to a minimum not considered here
    z = C1 * gamma['b'] * gamma['f'] * gamma['beta'] * gamma['xi'] * H_s
    # z_max =  gamma['f'] * gamma['beta'] * (C2 - C3)/np.sqrt(gamma['xi'])  *  H_s
    # van der meer :
    z_max = 3.2 * gamma['f'] * gamma['beta'] * H_s

    z = min(z, z_max)

    return z


