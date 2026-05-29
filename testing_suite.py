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
from Estimator import estimate_latent_and_input,tune_prior

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
    
    estimate_latent_and_input_testing(channel_pulse_data_set[0],
                                                      LatentDim=LATENT_DIM,
                                                      InputDim=INPUT_DIM,
                                                      sim=sim,
                                                      inputs= channel_pulse_signal,
                                                      unseen_trial=channel_pulse_data_set[2])
    

    # estimate_latent_and_input_testing(prbs_simulated_data_set[0],
    #                                                   LatentDim=LATENT_DIM,
    #                                                   InputDim=INPUT_DIM,
    #                                                   sim=sim,
    #                                                   inputs= prbs_signal,
    #                                                   unseen_trial=prbs_simulated_data_set[2])


    
    latent_state, input_est = estimate_latent_and_input(prbs_simulated_data_set[0], 5, 2)



    time = np.arange(prbs_simulated_data_set[0].shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    # 1. Observation (the raw neural data)
    axes[0].plot(time, channel_pulse_data_set[0], lw=0.8)
    axes[0].set_ylabel("y (observations)")
    axes[0].set_title("Observed neural activity (one line per neuron)")
    axes[0].grid(alpha=0.3)

    # 2. Recovered latent states
    axes[1].plot(time, latent_state, lw=1.0)
    axes[1].set_ylabel("x̂ (latent state)")
    axes[1].set_title("Recovered latent states")
    axes[1].grid(alpha=0.3)
    axes[1].legend([f"x̂[{i}]" for i in range(latent_state.shape[1])], fontsize=7, loc="best")

    # 3. Recovered input
    axes[2].plot(time, input_est, lw=1.0)
    axes[2].set_ylabel("û (recovered input)")
    axes[2].set_title("Recovered input")
    axes[2].set_xlabel("time step")
    axes[2].grid(alpha=0.3)
    axes[2].legend([f"û[{i}]" for i in range(input_est.shape[1])], fontsize=7, loc="best")

    plt.tight_layout()
    plt.show()    

testing_suite()