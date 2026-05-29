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
from Subspace_ID import Subspace_ID
import math



def estimate_latent_and_input_testing(observation:np.ndarray, 
                              LatentDim:int, 
                              InputDim:int,
                              sim:Simulator,
                              inputs:np.ndarray,
                              unseen_trial:np.ndarray,
                        
                              
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

 
    simulated_params = sim.params
    
   
    prbs_simulated_data_set =sim.generate_dataset(inputs=inputs,num_trials = 5,num_timesteps= observation.shape[0],use_initialisation_prior=True)
  
    
    em_prbs_params, em_prbs,em_angles,em_RMSE_arr,em_omega,em_mag_sim,em_mag_est   = Estimator_Analytics.em_estimation_analytics(
        observation=observation,
        LatentDim=LatentDim, 
        InputDim=InputDim,
        inputs=inputs,
        simulated_params=sim.params,
        unseen_trial=unseen_trial,
    )
    illustrator.plot_all_ssm_matrices(simulated_params=simulated_params,estimated_params=em_prbs_params)


    #testing Subspace id

    subspace_prbs_params, _,sub_angles,_sub_RMSE_arr,sub_omega,sub_mag_sim,sub_mag_est = Estimator_Analytics.subspace_id_analytics(observation = observation, 
                                                                                                         LatentDim=LatentDim,
                                                                                                         InputDim=InputDim,
                                                                                                         inputs = inputs,
                                                                                                         unseen_trial=unseen_trial,horizon = 5,simulated_params=sim.params,)
    illustrator.plot_all_ssm_matrices(simulated_params=simulated_params,estimated_params=subspace_prbs_params)

    
    #testing combined
    subspace_EM_prbs_params, _,subem_angles,subem_RMSE_arr,subem_omega,subem_prbs_mag_sim,subem_mag_est = Estimator_Analytics.subspace_em_analytics(observation = observation, 
                                                                                                         LatentDim=LatentDim,
                                                                                                         InputDim=InputDim,
                                                                                                         inputs = inputs,
                                                                                                         unseen_trial=unseen_trial,horizon = 5,simulated_params=sim.params,)

    illustrator.plot_all_ssm_matrices(simulated_params=simulated_params,estimated_params=subspace_EM_prbs_params)



    print('fini')

   # ============================== PLOTTING ==============================
    # Three estimators compared throughout: EM (random init), SSID alone, SSID+EM.
    # Ground truth shown where applicable.
    EM_COLOR, SSID_COLOR, SSIDEM_COLOR, TRUE_COLOR = 'C0', 'C3', 'C2', 'k'
    horizons = np.asarray([1, 2, 5, 10, 20, 30])   # must match _run_diagnostics

    # --- Eigenvalue comparison (complex plane) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    true_eig    = np.linalg.eigvals(sim.params.A)
    em_eig      = np.linalg.eigvals(em_prbs_params.A)
    ssid_eig    = np.linalg.eigvals(subspace_prbs_params.A)
    ssidem_eig  = np.linalg.eigvals(subspace_EM_prbs_params.A)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', lw=0.8, label='unit circle')
    ax.axhline(0, color='gray', lw=0.5, zorder=0)
    ax.axvline(0, color='gray', lw=0.5, zorder=0)
    ax.scatter(true_eig.real,   true_eig.imag,   marker='o', s=80,
            facecolors='none', edgecolors=TRUE_COLOR, label='true')
    ax.scatter(em_eig.real,     em_eig.imag,     marker='x', s=80,  color=EM_COLOR,     label='EM')
    ax.scatter(ssid_eig.real,   ssid_eig.imag,   marker='+', s=100, color=SSID_COLOR,   label='SSID')
    ax.scatter(ssidem_eig.real, ssidem_eig.imag, marker='*', s=100, color=SSIDEM_COLOR, label='SSID+EM')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('Re'); ax.set_ylabel('Im'); ax.set_title('Eigenvalues of A')
    ax.legend(); plt.tight_layout(); plt.show()

    # --- Subspace angles on C ---
    fig, ax = plt.subplots(figsize=(6, 3.5))
    dims  = np.arange(1, len(em_angles) + 1)
    width = 0.27
    ax.bar(dims - width, em_angles,     width, label='EM',      color=EM_COLOR)
    ax.bar(dims,         sub_angles,    width, label='SSID',    color=SSID_COLOR)
    ax.bar(dims + width, subem_angles,  width, label='SSID+EM', color=SSIDEM_COLOR)
    ax.axhline(5, ls='--', color='gray', lw=0.8)
    ax.set_xlabel('dim'); ax.set_ylabel('degrees')
    ax.set_title('C subspace angles')
    ax.set_xticks(dims); ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()

    # --- Multi-step RMSE (with unfitted baseline) ---
    random_params = sim.generate_general_ssm_matrices()
    RMSE_unfitted = sim.get_RMSE_array(em=em_prbs,
                                    est_params_dynamax=LDSParams.to_dynamax(random_params),
                                    y_trial=unseen_trial, inputs=inputs, k=horizons)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

    for ax, show_baseline in [(ax1, False), (ax2, True)]:
        ax.plot(horizons, em_RMSE_arr,    'o-',  color=EM_COLOR,     lw=1.2, label='EM')
        ax.plot(horizons, _sub_RMSE_arr,  's-',  color=SSID_COLOR,   lw=1.2, label='SSID')
        ax.plot(horizons, subem_RMSE_arr, '^-',  color=SSIDEM_COLOR, lw=1.2, label='SSID+EM')
        if show_baseline:
            ax.plot(horizons, RMSE_unfitted, 'd--', color='gray', lw=1.0, label='unfitted (random)')
        ax.set_xlabel('prediction horizon k')
        ax.set_title('Multi-step prediction RMSE' + (' with baseline' if show_baseline else ''))
        ax.legend()
        ax.grid(alpha=0.3)

    ax1.set_ylabel('RMSE')
    fig.tight_layout()
    plt.show()

    # --- Frequency response: per (output, input) cell, true / EM / SSID / SSID+EM overlaid --- 
    
    y_dim = observation.shape[1]
    u_dim = InputDim

    n_cols = 4
    n_rows_per_input = math.ceil(y_dim / n_cols)  # 4 for 16 outputs
    n_rows = u_dim * n_rows_per_input

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 20),
        sharex=True, squeeze=False
    )

    for j in range(u_dim):
        for i in range(y_dim):
            row = j * n_rows_per_input + (i // n_cols)
            col = i % n_cols
            ax = axes[row, col]

            ax.plot(em_omega,    em_mag_sim[:, i, j],    '-',  color=TRUE_COLOR,   lw=1.0, label='true')
            ax.plot(em_omega,    em_mag_est[:, i, j],    '--', color=EM_COLOR,     lw=0.8, label='EM')
            ax.plot(sub_omega,   sub_mag_est[:, i, j],   ':',  color=SSID_COLOR,   lw=0.8, label='SSID')
            ax.plot(subem_omega, subem_mag_est[:, i, j], '-.', color=SSIDEM_COLOR, lw=0.8, label='SSID+EM')

            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=6)
            ax.set_ylabel(f'y[{i}] (dB)', fontsize=6)
            if i // n_cols == 0:
                ax.set_title(f'y[{i}]', fontsize=7, pad=2)
            if i // n_cols == n_rows_per_input - 1:
                ax.set_xlabel('ω (rad/sample)', fontsize=6)

            if col == 0:
                ax.annotate(f'input {j}', xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-42, 0), textcoords='offset points',
                            fontsize=7, va='center', rotation=90)

    # hide unused trailing axes
    for idx in range(y_dim, n_rows_per_input * n_cols):
        for j in range(u_dim):
            axes[j * n_rows_per_input + idx // n_cols, idx % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=4,
            loc='lower center', bbox_to_anchor=(0.5, -0.01))

    fig.suptitle('Frequency response: true vs EM vs SSID vs SSID+EM', fontsize=9, y=1.0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.04, hspace=0.55, wspace=0.5)
    plt.show()