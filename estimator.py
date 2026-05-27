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




def estimate_latent_and_input(observation:np.ndarray, LatentDim:int, InputDim:int,
                               simulated_params=None,
                               save_load=None,
                               unseen_trial=None):
        
    '''
    Estimate the latent states and inputs from the observed neural activity.

    Parameters:
    - observation: A 2D array of shape (Timepoints, Neurons) representing the observed neural activity.
    - LatentDim: The dimensionality of the latent state space.
    - InputDim: The dimensionality of the input space.

    Returns:
    - A tuple containing:
        - latent_states: A 2D array of shape (Timepoints, LatentDim) representing the estimated latent states over time.
    '''

    sim = Simulator(LatentDim, InputDim)
    pulse = sim.make_pulse_array(0, 40, 20, InputDim,
                                total_signal_length=observation.shape[0])

    best_params, em = Estimator_Analytics.em_estimation_analytics(
        observation=observation,
        LatentDim=LatentDim, InputDim=InputDim,
        inputs=pulse,
        simulated_params=simulated_params,
        save_load=save_load,
        unseen_trial=unseen_trial,
    )

    # Use the fitted em.model, not a fresh one
    lgssm_params = LDSParams.to_dynamax(best_params)
    posterior = em.model.smoother(lgssm_params,
                                emissions=jnp.asarray(observation),
                                inputs=jnp.asarray(pulse))
    latent_states = np.asarray(posterior.smoothed_means)
    return latent_states, pulse