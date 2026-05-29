import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
from typing import Tuple
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.parameters import ParameterProperties

from LDSParams import LDSParams
from Subspace_ID import Subspace_ID

jax.config.update("jax_enable_x64", True)


class Subspace_and_EM_Blind:
    """
    EM fitting for a linear Gaussian SSM, initialised from N4SID rather than
    random draws. SSID gives a globally consistent, non-iterative estimate
    already in a good basin of the EM likelihood, so a single restart usually
    converges quickly and reliably. Extra restarts (small Gaussian
    perturbations around the SSID seed) are available for robustness checks.
    """

    def __init__(self,
                 chosen_state_dimension: int,
                 chosen_input_dimension: int,
                 observation: np.ndarray,
                 horizon: int,
                 phi:float = 0.9,
                 sigma_u_sq:float = 1.0,
                 num_em_iters: int = 100,
                 num_restarts: int = 1,
                 psd_jitter: float = 1e-6
               ):
        """
        Parameters
        ----------
        observation : (T, neuron_count)
        inputs : (T, input_dim)
        horizon : block-Hankel horizon i for N4SID; must exceed state_dim.
                  A safe default is max(state_dim + 2, 2 * state_dim).
        num_restarts : 1 = pure SSID-seeded run (default). >1 = SSID seed
                       plus perturbed restarts.
        perturbation_scale : std of additive Gaussian noise on (A, B, C) for
                             restarts > 0. Ignored when num_restarts == 1.
        psd_jitter : tiny diagonal added to Q, R for numerical PSD safety,
                     since finite-sample N4SID residual covariances can drift
                     off the PSD cone.
        """
        self.observation = observation
        self.phi = phi
        self.sigma_u_sq = sigma_u_sq


        self.state_dim = chosen_state_dimension   
        self.input_dim = chosen_input_dimension
        self.state_dim_true = chosen_state_dimension
        self.input_dim_true = chosen_input_dimension
        self.emission_dim = observation.shape[1]
        self.horizon = horizon
        self.num_em_iters = num_em_iters
        self.num_restarts = num_restarts

        self.psd_jitter = psd_jitter
  

        self.model = LinearGaussianSSM(
            state_dim=chosen_state_dimension+ chosen_input_dimension,   #here we are going to do state augmentation
            emission_dim=self.emission_dim,
            input_dim=0,
        )

        # Run SSID once up front; restarts perturb this fixed seed.
        self.ssid_seed: LDSParams = Subspace_ID.stochastic_n4sid(outputs=observation,
                                                                 latent_dim=chosen_state_dimension,
                                                                 horizon=horizon,
                                                                 )

    def _initialise_from_ssid(self, key, perturb: bool):
        """Build (params, props) from the SSID seed, optionally perturbed."""
        seed = self.ssid_seed
        # print(f"=== SEED ===")
        # print(f"A_seed eigs (abs): {np.abs(np.linalg.eigvals(np.asarray(seed.A)))}")
        # print(f"Q_seed eigvals:    {np.linalg.eigvalsh(0.5*(np.asarray(seed.Q)+np.asarray(seed.Q).T))}")
        # print(f"R_seed eigvals:    {np.linalg.eigvalsh(0.5*(np.asarray(seed.R)+np.asarray(seed.R).T))}")
        # print(f"Q_seed Frobenius:  {np.linalg.norm(seed.Q):.4f}")
        n, m, p = self.state_dim_true, self.input_dim_true, self.emission_dim

        # Original seed pieces (B, D start at zero since SSID didn't give them)
        A_seed = np.asarray(seed.A)                          # (n, n)
        C_seed = np.asarray(seed.C)                          # (p, n)
        B_init = np.zeros((n, m))
      
        
        Q_seed = 0.5 * (np.asarray(seed.Q) + np.asarray(seed.Q).T) + self.psd_jitter * np.eye(n)

        
        y = np.asarray(self.observation)
        R_seed = 0.1 * np.diag(y.var(axis=0))

        # AR(1) prior on the input
        Phi   = self.phi * np.eye(m)
        Q_u   = self.sigma_u_sq * np.eye(m)

        # Block matrices for the augmented LGSSM
        A_tilde = np.block([[A_seed, B_init],
                            [np.zeros((m, n)), Phi]])
        C_tilde = np.hstack([C_seed, np.zeros((p,m))])
        Q_tilde = np.block([[Q_seed,          np.zeros((n, m))],
                            [np.zeros((m, n)), Q_u]])
        R_tilde = R_seed
    
        params, props = self.model.initialize(
            key,
            initial_mean=jnp.zeros(n + m),
            initial_covariance=jnp.eye(n + m),  
            dynamics_weights=A_tilde,
            dynamics_covariance=jnp.asarray(Q_tilde),
            emission_weights=C_tilde,
            emission_covariance=jnp.asarray(R_tilde),
        )

        # Freeze initial-state stats, mirroring the random-init class.
        props = props._replace(
            initial=props.initial._replace(
                mean=ParameterProperties(trainable=False),
                cov=ParameterProperties(trainable=False,
                                        constrainer=props.initial.cov.constrainer),
            )
        )
        return params, props

    def fit(self) -> Tuple[LDSParams, float, np.ndarray,np.ndarray,np.ndarray]:
        """Run EM from the SSID seed and return (best_params, best_ll, ll_trace).
        
        
        returns
        -------
        bestparams:LDSParams , best_ll:float,  best_llr_trace:np.ndarray, estimated latent states, estimated inputs 
        
        
        """



        emissions = jnp.asarray(self.observation)
    
        best_params, best_ll = None, float("-inf")
        best_llr_trace = np.zeros(self.num_em_iters)

        bar = tqdm(range(self.num_restarts), desc="EM (SSID-init)")
        for restart in bar:
            key = jr.PRNGKey(restart)
            params, props = self._initialise_from_ssid(key, perturb=(restart > 0))

            params, llr = self.model.fit_em(
                params, props,
                emissions=emissions,
                num_iters=self.num_em_iters,
            )

            llr_local = np.asarray(llr)
            if np.any(~np.isfinite(llr_local)):
                tqdm.write(f"restart {restart}: diverged, skipping")
                continue

            final_llr = float(llr[-1])
            if final_llr > best_ll:
                best_ll = final_llr
                best_params = params
                best_llr_trace = llr_local

            bar.set_postfix(best_ll=f"{best_ll:.2f}", current_ll=f"{llr[-1]:.2f}")

        if best_params is None:
            raise RuntimeError("All EM restarts diverged from the SSID seed. "
                               "Check horizon and state_dim.")
        
       # 2. Run smoother on the AUGMENTED params to get x̂ and û.
        posterior = self.model.smoother(best_params, emissions=emissions)
        smoothed_mean = np.asarray(posterior.smoothed_means)   # (T, n+m)
        self.x_hat = smoothed_mean[:, :self.state_dim_true]
        self.u_hat = smoothed_mean[:, self.state_dim_true:]  # recovered input
                  

        # Also store the augmented params in case you want to re-smooth later
        self.fitted_augmented_params = best_params
        

        n, m, p = self.state_dim_true, self.input_dim_true, self.emission_dim
        A_full = np.asarray(best_params.dynamics.weights)        # (n+m, n+m)
        C_full = np.asarray(best_params.emissions.weights)       # (p, n+m)
        Q_full = np.asarray(best_params.dynamics.cov)
        R_full = np.asarray(best_params.emissions.cov)

        A_fit = A_full[:n, :n]
        B_fit = A_full[:n, n:]
        C_fit = C_full[:, :n]
       
        Q_fit = Q_full[:n, :n]
        R_fit = R_full

        # Return as your LDSParams type
        out = LDSParams(A=A_fit, B=B_fit, C=C_fit, Q=Q_fit, R=R_fit,
                        mu_0=np.zeros(n), P_0=np.eye(n))
        


       


        return out, best_ll, best_llr_trace, self.x_hat,self.u_hat