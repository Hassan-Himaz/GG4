import numpy as np
from Dynamax_EM_fitting import Dynamax_EM_Fitting
from cProfile import label
from jax import vmap
from matplotlib import figure
from typing import Tuple,Callable
from Subspace_ID import Subspace_ID
from LDSParams import LDSParams
from Illustrator import Illustrator
import matplotlib.pyplot as plt
from Simulator import Simulator
import jax.numpy as jnp
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import subspace_angles
from Subspace_and_EM import Subspace_and_EM

class Estimator_Analytics():

    @staticmethod
    def em_estimation_analytics(
        observation: np.ndarray,
        LatentDim: int,
        InputDim: int,
        unseen_trial: np.ndarray,
        inputs:np.ndarray,
        simulated_params: LDSParams,            # "save", "load", or None
        num_iter: int = 30,
        num_restarts: int = 25,
        run_name: str = "run_002",
    ) -> Tuple[LDSParams, "Dynamax_EM_Fitting",np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
        Fit (or load) an LGSSM via EM, then run diagnostic plots.
        Returns (best_params, fitted_em) so the caller can use the fitted model
        directly for smoothing/prediction without re-instantiating EM.

        parameters
        -----------


        '''
        illustrator = Illustrator(np.load("ExampleDataset.npy"))

        em = Dynamax_EM_Fitting(
            chosen_state_dimension=LatentDim,
            chosen_input_dimension=InputDim,
            observation=observation,
            num_em_iters=num_iter,
            num_restarts=num_restarts,
            inputs=inputs,

        )

        em_fitted_params ,best_ll,return_llr = em.fit()
        # Diagnostics only if we have ground truth to compare against
        illustrator.plot_ll(return_llr)
    
        angles,RMSE_arr,omega,mag_sim,mag_est = Estimator_Analytics._run_diagnostics(illustrator=illustrator,
                                                                                     em=em,
                                                                                     simulated_params=simulated_params,
                                                                                     inputs=inputs,
                                                                                     best_params=em_fitted_params,
                                                                                     LatentDim=LatentDim,
                                                                                     InputDim=InputDim,
                                                                                     unseen_trial=unseen_trial,


                                                                                     
                                                                                     )

        return em_fitted_params, em,angles,RMSE_arr,omega,mag_sim,mag_est
    
    @staticmethod
    def subspace_id_analytics(
                              observation: np.ndarray,
                              LatentDim: int,
                              InputDim: int,
                              inputs: np.ndarray,
                              unseen_trial: np.ndarray,
                              horizon:int,
                              simulated_params: LDSParams,
                              save_load: str | None = None,            # "save", "load", or None
                              num_iter: int = 30,
                              num_restarts: int = 50,
                              
                              run_name: str = "run_002",
                              ) -> Tuple[LDSParams, "Dynamax_EM_Fitting",np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        
        '''
        use to run the analytic metrics on subspace estimation method

        parameters
        ----------

        horizon:int 
            horizon used in the hankel matrices needs to less than half the trial data length -1  ie (T- 1)/2
        
        '''
  

        illustrator = Illustrator(np.load("ExampleDataset.npy"))

        em = Dynamax_EM_Fitting(                       #initialise an em object to use kalman filter
            chosen_state_dimension=LatentDim,
            chosen_input_dimension=InputDim,
            observation=observation,
            num_em_iters=num_iter,
            num_restarts=num_restarts,
            inputs=inputs,
        )


        #Here we use the subspace id method to fit the data

        best_params = Subspace_ID.n4sid(inputs = inputs,outputs=observation,latent_dim=LatentDim,horizon=horizon)

        # Diagnostics only if we have ground truth to compare against
      
        angles,RMSE_arr,omega,mag_sim,mag_est = Estimator_Analytics._run_diagnostics(illustrator, em, simulated_params,inputs = inputs, best_params = best_params,
                        LatentDim = LatentDim, InputDim=InputDim, unseen_trial= unseen_trial)

        return best_params, em,angles,RMSE_arr,omega,mag_sim,mag_est 
    
    @staticmethod
    def subspace_em_analytics(
        observation: np.ndarray,
        LatentDim: int,
        InputDim: int,
        inputs: np.ndarray,
        unseen_trial: np.ndarray,
        horizon: int,
        simulated_params: LDSParams,
        num_iter: int = 30,
        num_restarts: int = 1,
        perturbation_scale: float = 0.0,
    ) -> Tuple[LDSParams, Subspace_and_EM,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Analytics for SSID-initialised EM: N4SID gives the seed, EM refines.

        Parameters
        ----------
        horizon : int
            Block-Hankel horizon for the SSID seed.
        num_restarts : int
            EM runs from (perturbed) SSID seeds. Default 1.
        perturbation_scale : float
            Gaussian std on (A, B, C) for restarts > 0. Ignored if num_restarts == 1.
        '''
        illustrator = Illustrator(np.load("ExampleDataset.npy"))

        em = Subspace_and_EM(
            chosen_state_dimension=LatentDim,
            chosen_input_dimension=InputDim,
            observation=observation,
            inputs=inputs,
            horizon=horizon,
            num_em_iters=num_iter,
            num_restarts=num_restarts,
            perturbation_scale=perturbation_scale,
        )
          

        best_params, _, mean_llr_evolution = em.fit()
        illustrator.plot_ll(mean_llr_evolution)

        return (best_params, em,
                *Estimator_Analytics._run_diagnostics(
                    illustrator=illustrator, em =em, simulated_params=simulated_params,
                    inputs=inputs, best_params=best_params,
                    LatentDim=LatentDim, InputDim=InputDim, unseen_trial=unseen_trial,
                ))


    

    @staticmethod
    def _run_diagnostics(illustrator:Illustrator, em:Dynamax_EM_Fitting|Subspace_and_EM, simulated_params:LDSParams,inputs:np.ndarray, best_params:LDSParams,
                        LatentDim:int, InputDim:int, unseen_trial:np.ndarray) ->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    

        '''
        returns
        --------
        angles
        
        
        '''
        sim_sim = Simulator(LatentDim, InputDim); sim_sim.set_params(simulated_params)
        est_sim = Simulator(LatentDim, InputDim); est_sim.set_params(best_params)

        

        eig_true = sim_sim.calculate_eigen()[0]
        eig_est  = est_sim.calculate_eigen()[0]
        # illustrator.plot_eigenvalues(eig_true, eig_est)

        angles = np.degrees(subspace_angles(simulated_params.C, best_params.C))
        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.bar(range(1, len(angles)+1), angles)
        # ax.axhline(5, ls="--", color="gray", lw=0.8)
        # ax.set_xlabel("dim"); ax.set_ylabel("degrees")
        # ax.set_title(f"C subspace angles (max {angles.max():.1f}°)")
        # plt.show()
            
        RMSE_arr = est_sim.get_RMSE_array(em=em,est_params_dynamax=LDSParams.to_dynamax(best_params),y_trial=unseen_trial,inputs=inputs,k=np.asarray([1,2,5,10,20,30]))

        # illustrator.plot_frequency_response(sim_sim=sim_sim, est_sim=est_sim)
        omega,mag_sim,mag_est = est_sim.get_frequency_response_data(sim_sim=sim_sim,est_sim=est_sim)

        return angles,RMSE_arr,omega,mag_sim,mag_est