from cProfile import label
from jax import vmap
from matplotlib import figure
import numpy as np
from Dynamax_EM_fitting import dynamax_EM_Fitting
from typing import Tuple,Callable
from LDSParams import LDSParams
from Illustrator import Illustrator
import matplotlib.pyplot as plt
from Simulator import Simulator
import jax.numpy as jnp
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import subspace_angles




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
    # #am going to just run through different methods for infering the SSM matrices 
    illustrator = Illustrator(np.load("ExampleDataset.npy")) # to make use of various plotting tools

    # #start with EM
    em = dynamax_EM_Fitting(chosen_state_dimension=LatentDim,chosen_input_dimension=InputDim,observation=observation,num_em_iters=100,num_restarts=100)

    # #will need to to a study on different inputs
    # sim = Simulator(num_hidden_states=LatentDim, num_inputs=InputDim) # define sim to use our inputs factory
    # pulse = sim.make_pulse_array(0,40,20,InputDim,total_signal_length=observation.shape[0])
   


    # best_params,best_ll = em.fit(inputs = pulse) # at the moment with no input

    # best_params.save('run001')  #these were causing my em to diverge and i dont know why, was really weird
    # best_params = LDSParams.load('run001')

    save_path_sim = Path.cwd() / "LDSParams_Saves" / "run_002_sim"
    save_path_est = Path.cwd() / "LDSParams_Saves" / "run_002_est"
    save_path_dataset = Path.cwd() / "LDSParams_Saves" / "run_002_dataset"

    # best_params.save(save_path_est)
    # simulated_params.save(save_path_sim)
    # np.savez(save_path_dataset,dataset = observation, inputs = pulse)

    best_params = LDSParams.load(save_path_est)
    simulated_params = LDSParams.load(save_path_sim)
    with np.load(save_path_dataset.with_suffix(".npz")) as f:
        dataset = f['dataset']
        pulse = f['inputs']

    if simulated_params is not None and best_params is not None:
        illustrator.plot_all_ssm_matrices(simulated_params,best_params)

        #-------------eigenvalues--------------------------------------------------

        #we want to compare eigenvalues of the the two A matrices, should be similar invariant to the similarity transform
        eig_true = np.linalg.eigvals(simulated_params.A)
        eig_est  = np.linalg.eigvals(best_params.A)

        #-----------SVD of C matrices-------------------------

        # orthonormal basis for column space of C matrice --> can find SVD decomp
        #columns of U span the observation space
        #columns of V span the latent space
        C_true = simulated_params.C
        C_est = best_params.C
        #column space basis is made up of the left singular vectors
        U_true, _, _ = np.linalg.svd(C_true, full_matrices=False)
        U_est,  _, _ = np.linalg.svd(C_est,  full_matrices=False)

        #can take top 3 dominant singular vectors and project into 3D for plotting

        V_true = U_true[:,:3]
        V_est = U_est[:,:3]

        coords_true = V_true.T @ V_true   # identity by construction
        coords_est  = V_true.T @ V_est

        #-----------------------------------------------------
        #subspace angles
        # what subspace angles is doing is finding orthnormal basis for column space of C matrices using QR to find the subspace they span this is the same as their U matrices from SVD
        #then if we look at the product of our orthogonal bases M =  Qest ^T Qsim each matix entry is the cosine of the angle between each basis vector vector in bases
        # 
        # we want best alignmet between bases  not between these particular bases
        #  M = U E V^t       #E standing in for sigma
        # U rotates in estimated basis ; V rotates in simualted basis 
        # (Qest U)^T (QsimV) = U^T M V = E
        # 
        # 
        # arcos of singular values are the best aligned angles between the orthonoral bases



        angles = np.degrees(subspace_angles(simulated_params.C,best_params.C))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(range(1, len(angles) + 1), angles)
        ax.axhline(5, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("dim"); ax.set_ylabel("degrees")
        ax.set_title(f"C subspace angles  (max {angles.max():.1f}°)")
        plt.show()


        #-------------kalman filter error----------------------------


        #---------transfer function analysis-----------------------





        #plotting

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")


        origin = np.zeros(3)
        for i in range(3):
            ax.quiver(*origin, *coords_true[:, i],
                    color="C0", arrow_length_ratio=0.1, label="true" if i == 0 else None)
            ax.quiver(*origin, *coords_est[:, i],
                    color="C1", arrow_length_ratio=0.1, label="est" if i == 0 else None)

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("dir 1"); ax.set_ylabel("dir 2"); ax.set_zlabel("dir 3")
        ax.set_title("Top-3 basis of column(C): true vs estimated\n(coordinates in true's frame)")
        ax.legend()
        plt.tight_layout()
        plt.show()



        # Scatter plot, both should sit near the same points inside the unit circle
        plt.scatter(eig_true.real,eig_true.imag,label='true')
        plt.scatter(eig_est.real,eig_est.imag,label='est')
        plt.legend()
        plt.show()


 
    #then need to generate state data with best params
    #will build time series with noise?
    #can find states predicted by estimated mdel by runing a kalman smoother

    lgssm_params = LDSParams.to_dynamax(best_params)
    posterior = em.model.smoother(lgssm_params, emissions=jnp.asarray(observation),
                                    inputs=jnp.asarray(pulse))
    latent_states = np.asarray(posterior.smoothed_means)   # shape (T, state_dim)

    
    latent_states = np.asarray(latent_states)
    inputs = pulse

    return latent_states, inputs
