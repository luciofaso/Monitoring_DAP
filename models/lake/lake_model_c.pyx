cimport numpy as np
import numpy as np

from libc.math cimport sqrt

cimport numpy as np


#def lake_sim_step_wrap()

cpdef tuple lake_sim_step_c (float h_bar_tmin1,
                                  float h_wind_t,
                                        float q_ij_t,
                                              float q_lat_t,
                                                    float temperature_t,
                                                          float h_wz_t,
                                                            float h_target_t,
                                                                  float pump_capacity,
                                                                    float Delta_t,
                                                                          float K,
                                                                                float A,
                                                                                      float k_evap,
                                                                                            float k_dem):

    q_free = K * sqrt(2 * 9.8 * max(h_bar_tmin1 + h_wind_t - h_wz_t, 0) )
    if h_bar_tmin1 >= h_target_t:
        u_release = 1
    else:
        u_release = 0
    q_out = (q_free + pump_capacity) * u_release
    q_evaporation_t = k_evap * temperature_t
    q_demand_t = k_dem * temperature_t  # oversimplified function of demand, proportional to temperature
    q_supply_t = q_demand_t  # heroic assumption, supply always satisfied TODO set low level lake limit

    h_bar_t = h_bar_tmin1 + Delta_t / A * (q_ij_t + q_lat_t - q_out - q_supply_t - q_evaporation_t)

    return h_bar_t, q_out, q_demand_t, q_supply_t
