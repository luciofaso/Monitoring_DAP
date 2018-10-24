import os


#compile pyx file
os.system('python setup_lake.py build_ext --inplace')
import pyximport; pyximport.install() #pyximport.install(pyimport = True)

from lake_model_c import lake_sim_step_c

lake_sim_step_c()

