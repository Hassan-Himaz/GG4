import numpy as np
import matplotlib.pyplot as plt
from Dynamax_EM_fitting import dynamax_EM_Fitting
from Explorer import Explorer
from Estimator import estimate_latent_and_input
from PEM_framework import PEM_Framework
from Illustrator import Illustrator
from Simulator import Simulator
from PEM_framework import PEM_Framework
from LDSParams import LDSParams
from typing import Callable

#--------------------------------------------


##   we want this to showcase everything that our

    


#----------------------------------------------

def testing_suite():
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    # # illustrator.run_all()

    # illustrator.plot_PCA()

    sim = Simulator(num_hidden_states=5,num_inputs=0)
    params = sim.generate_general_ssm_matrices()
    sim.set_params(params)
    # sim.generate_and_show()
    simulated_data_set = sim.generate_dataset(num_trials = 5, num_timesteps=60,use_initialisation_prior=True)
    # simulated_data_illustrator = Illustrator(data_set)
    # simulated_data_illustrator.plot_heatmap
    # summary_dict = sim.gramian_summary()
    # print(summary_dict)

    latent_states, inputs = estimate_latent_and_input(simulated_data_set[0],LatentDim=5,InputDim=0,simulated_params=params)
    

    
testing_suite()