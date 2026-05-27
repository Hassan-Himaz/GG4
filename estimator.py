from cProfile import label

import numpy as np
from Dynamax_EM_fitting import dynamax_EM_Fitting
from typing import Tuple
from LDSParams import LDSParams
from Illustrator import Illustrator
import matplotlib.pyplot as plt



def estimate_latent_and_input(observation: np.ndarray, LatentDim: int, InputDim: int, simulated_params:LDSParams|None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the latent states and inputs from the observed neural activity.

    Parameters:
    - observation: A 2D array of shape (Timepoints, Neurons) representing the observed neural activity.
    - LatentDim: The dimensionality of the latent state space.
    - InputDim: The dimensionality of the input space.
    - simulated_params -> option to pass in simulated SSM matrices to visually compare

    Returns:
    - A tuple containing:
        - latent_states: A 2D array of shape (Timepoints, LatentDim) representing the estimated latent states over time.
        - inputs: A 2D array of shape (Timepoints, InputDim) representing the estimated inputs over time.
    """
    #am going to just run through different methods for infering the SSM matrices 
    illustrator = Illustrator(np.load("ExampleDataset.npy")) # to make use of various plotting tools

    #start with EM
    em = dynamax_EM_Fitting(chosen_state_dimension=LatentDim,chosen_input_dimension=InputDim,observation=observation,num_em_iters=100,num_restarts=100)
    best_params,best_ll = em.fit() # at the moment with no input
    if simulated_params is not None and best_params is not None:
        illustrator.plot_all_ssm_matrices(simulated_params,best_params)

        #we want to compare eigenvalues of the the two A matrices, should be similar invariant to the similarity transform
        eig_true = np.linalg.eigvals(simulated_params.A)
        eig_est  = np.linalg.eigvals(best_params.A)
        # Scatter plot, both should sit near the same points inside the unit circle
        plt.scatter(eig_true.real,eig_true.imag,label='true')
        plt.scatter(eig_est.real,eig_est.imag,label='est')
        plt.legend()
        plt.show()


 
    #then need to generate state data with best params
    #will build time series with noise?
    rng = np.random.default_rng()
    latent_state = rng.multivariate_normal(best_params.mu_0,best_params.P_0)
    observed_state = np.matmul(best_params.C,latent_state) + rng.multivariate_normal(np.zeros(observation.shape[1]),best_params.R)
    _, latent_states = [observed_state], [latent_state]

    for time in range(observation.shape[0]):
        latent_state = np.matmul(best_params.A,latent_state) + rng.multivariate_normal(np.zeros(LatentDim),best_params.Q)
        observed_state = np.matmul(best_params.C,latent_state) + rng.multivariate_normal(np.zeros(observation.shape[1]),best_params.R)
        latent_states.append(latent_state)

    
    latent_states = np.asarray(latent_states)
    inputs = np.zeros(observation.shape[0])
    
    return latent_states, inputs
