# Model adapted from xx
# Full information to be found in
# Wave run-up and overtopping Jentsje W. van der Meer

cimport numpy as np
import numpy as np

#from scipy.stats import gumbel_r, norm


from libc.math cimport sqrt, log, exp


cdef float g = 9.8  # gravity accel m^2/s




cdef tuple hydraulic_conditions(float SWL):
    """ calculation of hydraulic conditions, considering xxx asyntoptic behaviour and stationarity of relationships.

    :param SWL: surge storm water level [m]
    :return: H_s, wave characteristics [m], Tm, spectral wave period [s]
    """

    cdef:
        float C1 = 2.70
        float C2 = -1.70
        float C3 = 0.1

    H_s = C1 * log(SWL) + C2
    Tm = sqrt(H_s / C3)

    return H_s,Tm




cdef double cond_prob_dike_failure(np.ndarray[np.float64_t,ndim=1] q_occ, float q_critical):
    """

    :param q_occ: occurring overtopping discharge
    :param q_critical: critical dike discharge
    :return:
    """
    # to add probabilistic evaluation, make m_o and m_c stochastic
    cdef:
        int i, N
        double m_o = 1
        double m_c = 1
        double z, Z

    N=len(q_occ)
    z = 0
    for i in range(N):
        if m_o * q_occ[i] > m_c * q_critical:
            z += 1
    Z = z / N
    return Z




def frequency_failure(average_water_level, surge_frequency, crown_height,slope,gamma_b,gamma_beta,gamma_f,q_critical):
    """

    :param average_water_level:
    :param surge_frequency:
    :param dike:
    :param gamma:
    :param q_critical:
    :return:
    """

    # raw implementation of importance sampling
    base_year=2000
    N = 10000
    surge_sample = surge_frequency.rvs(N * base_year)
    surge_base_year = surge_frequency.ppf(1 - 1 / base_year)  #NOT WORKING IN CYTHON
    surge_importance_sample = surge_sample[surge_sample >= surge_base_year]
    freq_failure=freq_fail_c(average_water_level, surge_importance_sample, crown_height, slope, gamma_b, gamma_beta, gamma_f, q_critical, N, base_year)

    return freq_failure




def freq_fail_c (float average_water_level,np.ndarray[np.double_t,ndim=1] surge_importance_sample,
                                                            float crown_height,
                                                              float slope,
                                                                    float gamma_b,
                                                                          float gamma_beta,
                                                                                float gamma_f,
                                                                                      float q_critical, float N, int base_year):


    cdef:
        int i, N_imp_sample
        int M = 100
        double surge_level, n_failure, prob_of_failure, frequency_of_failure
        np.ndarray[np.double_t, ndim=1] q_occ


    N_imp_sample = len(surge_importance_sample)
    n_failure = 0
    for i in range(N_imp_sample):
        surge_level = surge_importance_sample[i]
        q_occ = occourred_discharge(average_water_level, surge_level, crown_height, slope, gamma_b, gamma_beta, gamma_f, M)
        n_failure += cond_prob_dike_failure(q_occ, q_critical)

    prob_of_failure = n_failure / ( N * base_year )
    #if prob_of_failure==0:
    frequency_of_failure = 1 / prob_of_failure

    return frequency_of_failure



cdef np.ndarray occourred_discharge(float average_water_level,
                              float surge_level,
                                    float crown_height,
                                          float slope,
                                                float gamma_b,
                                                      float gamma_beta,
                                                            float gamma_f,
                                                                  int N):
#cdef np.ndarray[double,ndim=1] occourred_discharge(float average_water_level, float surge_level, Dike dike, Gamma gamma, int N):
    """ sample of occurred discharges, given a water lavel, surge level and dike carachteristics

    :param average_water_level: average long term water level
    :param surge_level: water level at storm, relative to average water level
    :param dike: dike caractheristics
    :param gamma: parameters, as defined in ref
    :param N: sample dimension
    :return:
    """
    cdef:
        int i
        double H_s, Tm, R_c, C4,C5,Q_b,Q_n
        np.ndarray[double, ndim = 1] q_occ = np.empty(N)

    H_s, Tm = hydraulic_conditions(surge_level)
    gamma_xi = relative_wave_runup(H_s, Tm, slope)
    R_c = crown_height - (average_water_level + surge_level)  # creast free board

    # discharge_breaking_waves
    for i in range(N):
        C4 = np.random.normal(4.7, 0.55)  # or 5.2
        Q_b = 0.067 / sqrt(slope) * gamma_b * gamma_xi * \
              exp(- C4 * R_c / (H_s * gamma_xi * gamma_b * gamma_f * gamma_beta))
        # discharge_non_breaking_waves
        C5 = np.random.normal(2.3, 0.35)  # or 2.6
        Q_n = 0.2 * exp(- C5 * R_c / (H_s * gamma_f * gamma_beta))
        q_occ[i] = min(Q_b, Q_n) * sqrt(g * H_s ** 3)  # m^3 /m / sec

    return (q_occ)




cdef float relative_wave_runup(float H_s, float T_s, float slope):
    """calculate relative wave runup from storm and dike characteristics

    :param H_s: Wave characteristics [m]
    :param T_s: Spectral wave period [s]
    :param slope: dike average slope [-]
    :return: xi, relative wave runup [m]
    """


    cdef float s_0, xi

    # s_0 is wave stepness
    s_0 = (2 * 3.14 * H_s) / (g * T_s ** 2)
    xi = slope / sqrt(s_0)

    return xi








