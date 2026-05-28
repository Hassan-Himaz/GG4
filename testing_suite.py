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


    LATENT_DIM = 5
    INPUT_DIM = 2
    TOTAL_TRIAL_LENGTH = 60

    
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    sim = Simulator(num_hidden_states=5,num_inputs=2)
    save_path_sim = Path.cwd() / "LDSParams_Saves" / "run_002_sim"
    params = LDSParams.load(save_path_sim)
  
    sim.set_params(params)
    
    
    # prbs_signal = sim.make_prbs_array(20,INPUT_DIM,total_signal_length=TOTAL_TRIAL_LENGTH)
    # prbs_simulated_data_set =sim.generate_dataset(inputs=prbs_signal,num_trials = 5,num_timesteps= TOTAL_TRIAL_LENGTH,use_initialisation_prior=True)

    ramp_signal = sim.make_ramp_array(t_on=0,t_off=40,slope=1.2,num_inputs=INPUT_DIM,total_signal_length=TOTAL_TRIAL_LENGTH)
    ramp_simulated_data_set =sim.generate_dataset(inputs=ramp_signal,num_trials = 5,num_timesteps= TOTAL_TRIAL_LENGTH,use_initialisation_prior=True)

    latent_states, inputs = estimate_latent_and_input(ramp_simulated_data_set[0],
                                                      LatentDim=LATENT_DIM,
                                                      InputDim=INPUT_DIM,
                                                      sim=sim,
                                                      save_load= None,
                                                      unseen_trial=ramp_simulated_data_set[2])

    
testing_suite()