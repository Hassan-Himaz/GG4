from cProfile import label
from jax import vmap
from matplotlib import figure
import numpy as np
from Dynamax_EM_fitting import Dynamax_EM_Fitting
from typing import Tuple,Callable
from Subspace_and_EM_Blind import Subspace_and_EM_Blind
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
from tqdm import tqdm
from itertools import product


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

    #auto tuning
    prior_hyper_parameter_dict = tune_prior(observation=observation,LatentDim=LatentDim,InputDim=InputDim)
    best_phi = prior_hyper_parameter_dict['phi']
    best_sigma_u_sq= prior_hyper_parameter_dict['sigma']
    estimator = Subspace_and_EM_Blind(chosen_state_dimension=LatentDim,chosen_input_dimension=InputDim,observation=observation,horizon=horizon,phi=best_phi,sigma_u_sq=best_sigma_u_sq)
    _,_,_,latent_states,inputs = estimator.fit()

    
    return latent_states, inputs


def tune_prior(observation: np.ndarray,
               LatentDim: int,
               InputDim: int,
               phis: list = [0.5, 0.9, 0.99],
               sigmas: list = [1, 100, 1000,10000,100000]) -> dict:
    """
    parameter sweep to tune prior

    print course grid

    returns
    --------
    best parameter settings
    """
    T = observation.shape[0]
    horizon = max(LatentDim + 2, min(2 * LatentDim, (T - 1) // 4))

    results = []
    grid = list(product(phis, sigmas))
    bar = tqdm(grid, desc="prior sweep")

    for phi, sigma in bar:
        bar.set_postfix(phi=f"{phi:.2f}", sigma=f"{sigma:.1f}")
        try:
            est = Subspace_and_EM_Blind(
                chosen_state_dimension=LatentDim,
                chosen_input_dimension=InputDim,
                observation=observation,
                horizon=horizon,
                phi=phi,
                sigma_u_sq=sigma,
            )
            _, _, _, x_hat, u_hat = est.fit()

            A_tilde = np.asarray(est.fitted_augmented_params.dynamics.weights)
            C_tilde = np.asarray(est.fitted_augmented_params.emissions.weights)
            xu = np.hstack([x_hat, u_hat])
            xu_pred = xu[:-1] @ A_tilde.T
            y_pred = xu_pred @ C_tilde.T

            rmse = float(np.sqrt(np.mean((observation[1:] - y_pred) ** 2)))
            results.append({'phi': phi, 'sigma': sigma, 'rmse': rmse})
            tqdm.write(f"phi={phi:5.2f}  sigma={sigma:7.1f}  RMSE={rmse:.4f}")
        except RuntimeError:
            tqdm.write(f"phi={phi:5.2f}  sigma={sigma:7.1f}  diverged")
            results.append({'phi': phi, 'sigma': sigma, 'rmse': np.inf})

    best = min(results, key=lambda r: r['rmse'])
    print(f"\nBest: phi={best['phi']}, sigma={best['sigma']}, RMSE={best['rmse']:.4f}")
    return best