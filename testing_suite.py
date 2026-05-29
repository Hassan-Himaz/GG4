import signal

import numpy as np
import matplotlib.pyplot as plt
from Illustrator import Illustrator
from Simulator import Simulator
from Explorer import Explorer
from Estimator_testing import estimate_latent_and_input_testing
from PEM_framework import PEM_Framework
from LDSParams import LDSParams
from typing import Callable
from pathlib import Path
from Dynamax_EM_fitting import Dynamax_EM_Fitting

def testing_suite():


    LATENT_DIM = 5
    INPUT_DIM = 2
    TOTAL_TRIAL_LENGTH = 120

    
    illustrator = Illustrator(np.load("ExampleDataset.npy"))
    sim = Simulator(num_hidden_states=5,num_inputs=2)
    save_path_sim = Path.cwd() / "LDSParams_Saves" / "run_002_sim"
    params = LDSParams.load(save_path_sim)
  
    sim.set_params(params)
    print(sim.params)
    
    
    prbs_signal = sim.make_prbs_array(20,INPUT_DIM,total_signal_length=TOTAL_TRIAL_LENGTH)
    prbs_simulated_data_set =sim.generate_dataset(inputs=prbs_signal,num_trials = 5,num_timesteps= TOTAL_TRIAL_LENGTH,use_initialisation_prior=True)

    channel_pulse_signal = sim.make_pulse_array_per_channel(np.asarray([0,0]),np.asarray([10,20]),np.asarray([50,25]),TOTAL_TRIAL_LENGTH,INPUT_DIM)
    channel_pulse_data_set =sim.generate_dataset(inputs=channel_pulse_signal,num_trials = 5,num_timesteps= TOTAL_TRIAL_LENGTH,use_initialisation_prior=True)

    # estimate_latent_and_input_testing(channel_pulse_data_set[0],
    #                                                   LatentDim=LATENT_DIM,
    #                                                   InputDim=INPUT_DIM,
    #                                                   sim=sim,
    #                                                   inputs= prbs_signal,
    #                                                   unseen_trial=channel_pulse_data_set[2])
    
    # estimate_latent_and_input_testing(channel_pulse_data_set[0],
    #                                                   LatentDim=LATENT_DIM,
    #                                                   InputDim=INPUT_DIM,
    #                                                   sim=sim,
    #                                                   inputs= channel_pulse_signal,
    #                                                   unseen_trial=channel_pulse_data_set[2])
    

    estimate_latent_and_input_testing(prbs_simulated_data_set[0],
                                                      LatentDim=LATENT_DIM,
                                                      InputDim=INPUT_DIM,
                                                      sim=sim,
                                                      inputs= prbs_signal,
                                                      unseen_trial=prbs_simulated_data_set[2])


    
testing_suite()