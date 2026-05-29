from cProfile import label
from jax import vmap
from matplotlib import figure
import numpy as np
from Dynamax_EM_fitting import Dynamax_EM_Fitting
from typing import Tuple,Callable
from GG4.Subspace_and_EM_Blind import Subspace_and_EM_Blind
from LDSParams import LDSParams
from Illustrator import Illustrator
import matplotlib.pyplot as plt
from Simulator import Simulator
import jax.numpy as jnp
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import subspace_angles
from Estimator_Analytics import Estimator_Analytics
from Subspace_ID import Subspace_ID
import math


def estimate_latent_and_input(observation: np.ndarray, LatentDim: int, InputDim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the latent states and inputs from the observed neural activity.

    Parameters:
    - observation: A 2D array of shape (Timepoints, Neurons) representing the observed neural activity.
    - LatentDim: The dimensionality of the latent state space.
    - InputDim: The dimensionality of the input space.

    Returns:
    - A tuple containing:
        - latent_states: A 2D array of shape (Timepoints, LatentDim) representing the estimated latent states over time.
        - inputs: A 2D array of shape (Timepoints, InputDim) representing the estimated inputs over time.
    """
    horizon = min(2 * LatentDim, (observation.shape[0] - 1) // 4)  #lower bound hankel window length has to be greater than latent dim cannot be larger than t-1/2
    horizon = max(horizon, LatentDim + 2)   # safety floor

    estimator = Subspace_and_EM_Blind(chosen_state_dimension=LatentDim,chosen_input_dimension=InputDim,observation=observation,horizon=horizon,)
    best_params = estimator.fit()

    # After EM converges, run the smoother to get the augmented state trajectory
    posterior = self.model.smoother(best_params, emissions=emissions)
    smoothed_mean = np.asarray(posterior.smoothed_means)   # (T, n+m)
    self.x_hat = smoothed_mean[:, :n]                       # latent state estimate
    self.u_hat = smoothed_mean[:, n:]                       # recovered input

    # Also store the augmented params in case you want to re-smooth later
    self.fitted_augmented_params = best_params


    # Placeholder implementation. Replace with actual estimation logic.
    Timepoints = observation.shape[0]
    latent_states = np.random.rand(Timepoints, LatentDim)
    inputs = np.random.rand(Timepoints, InputDim)
    
    return latent_states, inputs