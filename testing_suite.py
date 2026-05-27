import numpy as np
import matplotlib.pyplot as plt
from Illustrator import Illustrator
from Simulator import Simulator
from Explorer import Explorer
from Estimator import estimate_latent_and_input
from PEM_framework import PEM_Framework
from LDSParams import LDSParams
from typing import Callable
from pathlib import Path
from Dynamax_EM_fitting import Dynamax_EM_Fitting

def testing_suite():
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    # # illustrator.run_all()

    # illustrator.plot_PCA()

    sim = Simulator(num_hidden_states=5,num_inputs=2)
    # params = sim.generate_general_ssm_matrices()
    save_path_sim = Path.cwd() / "LDSParams_Saves" / "run_002_sim"
    params = LDSParams.load(save_path_sim)
    sim.set_params(params)

    simulated_data_set = sim.generate_dataset(num_trials = 5, num_timesteps=60,use_initialisation_prior=True)


    latent_states, inputs = estimate_latent_and_input(simulated_data_set[0],
                                                      LatentDim=5,
                                                      InputDim=2,
                                                      simulated_params=params,
                                                      save=False,
                                                      load=True,
                                                      unseen_trial=simulated_data_set[2])

    
testing_suite()