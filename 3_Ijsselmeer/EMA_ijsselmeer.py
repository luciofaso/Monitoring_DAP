import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.ijssel_system import test_ijsselmeer_cc
from ema_workbench import RealParameter, CategoricalParameter, ScalarOutcome, Model
from ema_workbench.analysis import prim, cart
import time
np.warnings.filterwarnings('ignore')

# set model
model = Model('Ijsselmeer', function=test_ijsselmeer_cc)

# set uncertainties
def distributed_sampling(x_min, x_max , num:int=10, distrib=np.log):
    """Create tuple of non homogeneous distributed sampling in a given range

    Args:
        x_min:
        x_max:
        num:
        distrib:

    Returns:

    """
    
    last_quantile = (num - 1) / num
    lambda_val = - distrib(1- (last_quantile) ) / (x_max - x_min)
    x_distributed = tuple(- distrib(1 - np.arange(0, 1, 1/ num)) / lambda_val + x_min)

    assert isinstance(x_distributed, tuple),  "is not tuple"
    return x_distributed

non_uniform_sampling = False

# set uncertainties
n_eval_per_var = 12
sampling = distributed_sampling if non_uniform_sampling == True else lambda x,y: (np.linspace(x, y, num=n_eval_per_var))



model.uncertainties  = [   #CategoricalParameter('wind_multiplier', sampling(1.0, 2.0) ),  # up to 100% increase in wind speed
                    CategoricalParameter('sea_level_shift', sampling(0.0, 1) ), # up to 1.5 meter shift water level
                    CategoricalParameter('peak_inflow_multiplier', sampling(1, 2) ), # up to 100% increase in inflow
                    CategoricalParameter('low_flow_shift',  sampling(0., 100.) ), # up to a reduction of low flow of 100 m^3/s
                    #CategoricalParameter('evaporation_multiplier', sampling(1, 2)),  # Increase of potential evaporation
                    #CategoricalParameter('precipitation_multiplier', sampling(1,3) )
                        ]

# set levers
policies =      [CategoricalParameter('summer_target', (-0.2, 0.2) ),
                CategoricalParameter('pump_capacity', (0, 500) ),
                #CategoricalParameter('winter_target', (-0.4, -0.1) ),
                 CategoricalParameter('sluices_widening', (1, 2) )
                 ]

model.levers = policies

# set outcomes
model.outcomes = [ScalarOutcome('dike failure frequency'),
                  ScalarOutcome('demand deficit relative')]

# run EMA
from ema_workbench import MultiprocessingEvaluator, ema_logging
ema_logging.log_to_stderr(ema_logging.INFO)





t = time.time()
with MultiprocessingEvaluator(model,n_processes=4) as evaluator:
    results = evaluator.perform_experiments(scenarios=n_eval_per_var**len(model.uncertainties),
                                            policies=1,levers_sampling=u'ff',uncertainty_sampling=u'ff')
elapsed = time.time() - t

print('tot calc time')
print(elapsed)
print('average calc time per experiment')
print(elapsed/len(results[0]))
experiments, outcomes = results

# save results
experiments_df = pd.DataFrame(experiments)
outcomes_df = pd.DataFrame(outcomes)
results_df = experiments_df.join(outcomes_df)
results_df.to_csv('../data/Ijsselmeer/2018_10_18_results.csv')

from ema_workbench.analysis import plotting,pairs_plotting

#fig, axes = pairs_plotting.pairs_scatter(results, group_by='policy')
#plotting.kde_over_time(results, outcomes_to_show=[], group_by=None, grouping_specifiers=None, results_to_show=None, colormap=u'coolwarm', log=True)

#from ema_workbench import save_results
#save_results(results, '2018_10_16_results.tar.gz')
