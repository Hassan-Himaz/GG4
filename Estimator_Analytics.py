import numpy as np
from Dynamax_EM_fitting import Dynamax_EM_Fitting
from cProfile import label
from jax import vmap
from matplotlib import figure
from typing import Tuple,Callable
from LDSParams import LDSParams
from Illustrator import Illustrator
import matplotlib.pyplot as plt
from Simulator import Simulator
import jax.numpy as jnp
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import subspace_angles

class Estimator_Analytics():

    @staticmethod
    def em_estimation_analytics(
        observation: np.ndarray,
        LatentDim: int,
        InputDim: int,
        inputs: np.ndarray,
        simulated_params: LDSParams | None = None,
        save_load: str | None = None,            # "save", "load", or None
        num_iter: int = 30,
        num_restarts: int = 200,
        unseen_trial: np.ndarray | None = None,
        run_name: str = "run_002",
    ) -> tuple[LDSParams, "Dynamax_EM_Fitting"] | None:
        '''
        Fit (or load) an LGSSM via EM, then run diagnostic plots.
        Returns (best_params, fitted_em) so the caller can use the fitted model
        directly for smoothing/prediction without re-instantiating EM.
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

        save_dir = Path.cwd() / "LDSParams_Saves"
        est_path = save_dir / f"{run_name}_est"
        sim_path = save_dir / f"{run_name}_sim"
        ds_path  = save_dir / f"{run_name}_dataset"

        if save_load == "load":
            best_params = LDSParams.load(est_path)
            if simulated_params is None:
                simulated_params = LDSParams.load(sim_path)
        else:
            best_params, best_ll, mean_llr_evolution = em.fit()
            illustrator.plot_ll(mean_llr_evolution)

            if save_load == "save":
                best_params.save(est_path)
                if simulated_params is not None:
                    simulated_params.save(sim_path)
                np.savez(ds_path.with_suffix(".npz"),
                        dataset=observation, inputs=inputs)

        # Diagnostics only if we have ground truth to compare against
        if simulated_params is not None:
            Estimator_Analytics._run_diagnostics(illustrator, em, simulated_params, best_params,
                            LatentDim, InputDim, unseen_trial)

        return best_params, em

    @staticmethod
    def _run_diagnostics(illustrator, em, simulated_params, best_params,
                        LatentDim, InputDim, unseen_trial):
        sim_sim = Simulator(LatentDim, InputDim); sim_sim.set_params(simulated_params)
        est_sim = Simulator(LatentDim, InputDim); est_sim.set_params(best_params)

        illustrator.plot_all_ssm_matrices(simulated_params, best_params)

        eig_true = sim_sim.calculate_eigen()[0]
        eig_est  = est_sim.calculate_eigen()[0]
        illustrator.plot_eigenvalues(eig_true, eig_est)

        angles = np.degrees(subspace_angles(simulated_params.C, best_params.C))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(range(1, len(angles)+1), angles)
        ax.axhline(5, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("dim"); ax.set_ylabel("degrees")
        ax.set_title(f"C subspace angles (max {angles.max():.1f}°)")
        plt.show()

        if unseen_trial is not None:
            y_pred = est_sim.kalman_filter_predicted_observations(em, unseen_trial)
            illustrator.plot_predicted_vs_actual(unseen_trial, y_pred, sim_sim.R)
            illustrator.plot_horizons_vs_RMSE(em, LDSParams.to_dynamax(best_params),
                                            unseen_trial, em.inputs, label='fitted')
            random_params = sim_sim.generate_general_ssm_matrices()
            illustrator.plot_horizons_vs_RMSE(em, LDSParams.to_dynamax(random_params),
                                            unseen_trial, em.inputs, label='not fit')

        illustrator.plot_frequency_response(sim_sim=sim_sim, est_sim=est_sim)