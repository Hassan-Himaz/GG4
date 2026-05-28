from cProfile import label
from jax import vmap
from matplotlib import figure
import numpy as np
from Dynamax_EM_fitting import Dynamax_EM_Fitting
from typing import Tuple,Callable
from LDSParams import LDSParams
from Illustrator import Illustrator
import matplotlib.pyplot as plt
from Simulator import Simulator
import jax.numpy as jnp
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import subspace_angles
from Estimator_Analytics import Estimator_Analytics




def estimate_latent_and_input(observation:np.ndarray, 
                              LatentDim:int, 
                              InputDim:int,
                              sim:Simulator|None = None,
                              save_load=None,
                              unseen_trial=None,
                                    
                              ):
        
    '''
    Estimate the latent states and inputs from the observed neural activity.

    Parameters:
    - observation: A 2D array of shape (Timepoints, Neurons) representing the observed neural activity.
    - LatentDim: The dimensionality of the latent state space.
    - InputDim: The dimensionality of the input space.

    simulator:Simulator|None = None,


    Returns:
    - A tuple containing:
        - latent_states: A 2D array of shape (Timepoints, LatentDim) representing the estimated latent states over time.
    '''

    illustrator = Illustrator(np.load("ExampleDataset.npy"))

    if sim is not None:
        simulated_params = sim.params

    else:
        
        sim = Simulator(num_hidden_states=5,num_inputs=2)
        save_path_sim = Path.cwd() / "LDSParams_Saves" / "run_002_sim"
        params = LDSParams.load(save_path_sim)
        sim.set_params(params)
   

    # going to need my estimate to generate inputs for a load of different inputs then see which one gives best results on data

    pulse = sim.make_pulse_array(0, 40, 20, InputDim,
                                total_signal_length=observation.shape[0])
    pulse_simulated_data_set =sim.generate_dataset(inputs=pulse,num_trials = 5, num_timesteps=observation.shape[0],use_initialisation_prior=True)
    

    ramp = sim.make_ramp_array(0,40,1.2,InputDim,total_signal_length=observation.shape[0])
    ramp_simulated_data_set =sim.generate_dataset(inputs=ramp,num_trials = 5, num_timesteps=observation.shape[0],use_initialisation_prior=True)
    
    channel_pulse = sim.make_pulse_array_per_channel(np.zeros(InputDim),
                                                     40*np.ones(InputDim),
                                                     np.asarray([30,20,10,40,5]),
                                                     total_signal_length=observation.shape[0], 
                                                     num_inputs=InputDim)
    channel_pulse_simulated_data_set =sim.generate_dataset(inputs=channel_pulse,num_trials = 5,num_timesteps= observation.shape[0],use_initialisation_prior=True)

    
    
    prbs_signal = sim.make_prbs_array(20,InputDim,total_signal_length=observation.shape[0])
    prbs_simulated_data_set =sim.generate_dataset(inputs=prbs_signal,num_trials = 5,num_timesteps= observation.shape[0],use_initialisation_prior=True)
  
    

    # pulse_best_params, em_pulse = Estimator_Analytics.em_estimation_analytics(
    #     observation=observation,
    #     LatentDim=LatentDim, InputDim=InputDim,
    #     inputs=pulse,
    #     simulated_params=simulated_params,
    #     save_load=save_load,
    #     unseen_trial=unseen_trial,
    # )

    ramp_best_params, em_ramp = Estimator_Analytics.em_estimation_analytics(
        observation=observation,
        LatentDim=LatentDim, InputDim=InputDim,
        inputs=ramp,
        simulated_params=simulated_params,
        save_load=save_load,
        unseen_trial=unseen_trial,
    )

    # channel_pulse_params, em_channel_pulse = Estimator_Analytics.em_estimation_analytics(
    #     observation=observation,
    #     LatentDim=LatentDim, InputDim=InputDim,
    #     inputs=channel_pulse,
    #     simulated_params=simulated_params,
    #     save_load=save_load,
    #     unseen_trial=unseen_trial,
    # )
    
    # prbs_params, em_prbs = Estimator_Analytics.em_estimation_analytics(
    #     observation=observation,
    #     LatentDim=LatentDim, InputDim=InputDim,
    #     inputs=prbs_signal,
    #     simulated_params=sim.params,
    #     save_load=save_load,
    #     unseen_trial=unseen_trial,
    # )



    # Use the fitted em.model, not a fresh one
    lgssm_params = LDSParams.to_dynamax(ramp_best_params)
    posterior = em_ramp.model.smoother(lgssm_params,
                                emissions=jnp.asarray(observation),
                                inputs=jnp.asarray(prbs_signal))
    latent_states = np.asarray(posterior.smoothed_means)
    return latent_states, prbs_signal