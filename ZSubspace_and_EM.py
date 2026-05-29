import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
from typing import Tuple
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.parameters import ParameterProperties

from ZLDSParams import LDSParams
from ZSubspace_ID import Subspace_ID

jax.config.update("jax_enable_x64", True)


class Subspace_and_EM:
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
                 inputs: np.ndarray,
                 horizon: int,
                 num_em_iters: int = 100,
                 num_restarts: int = 1,
                 perturbation_scale: float = 0.0,
                 psd_jitter: float = 1e-6):
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
        self.inputs = inputs
        self.state_dim = chosen_state_dimension
        self.input_dim = chosen_input_dimension
        self.emission_dim = observation.shape[1]
        self.horizon = horizon
        self.num_em_iters = num_em_iters
        self.num_restarts = num_restarts
        self.perturbation_scale = perturbation_scale
        self.psd_jitter = psd_jitter

        self.model = LinearGaussianSSM(
            state_dim=chosen_state_dimension,
            emission_dim=self.emission_dim,
            input_dim=chosen_input_dimension,
        )

        # Run SSID once up front; restarts perturb this fixed seed.
        self.ssid_seed: LDSParams = Subspace_ID.n4sid(
            inputs=inputs,
            outputs=observation,
            latent_dim=chosen_state_dimension,
            horizon=horizon,
        )

    def _initialise_from_ssid(self, key, perturb: bool):
        """Build (params, props) from the SSID seed, optionally perturbed."""
        seed = self.ssid_seed
     
        n, m, p = self.state_dim, self.input_dim, self.emission_dim
        key_pert, key_init = jr.split(key)

        if perturb and self.perturbation_scale > 0:
            kA, kB, kC = jr.split(key_pert, 3)
            A = jnp.asarray(seed.A) + self.perturbation_scale * jr.normal(kA, (n, n))
            B = jnp.asarray(seed.B) + self.perturbation_scale * jr.normal(kB, (n, m))
            C = jnp.asarray(seed.C) + self.perturbation_scale * jr.normal(kC, (p, n))
        else:
            A = jnp.asarray(seed.A)
            B = jnp.asarray(seed.B)
            C = jnp.asarray(seed.C)

        # Symmetrise + jitter to keep Q, R strictly PSD.
        Q = 0.5 * (seed.Q + seed.Q.T) + self.psd_jitter * np.eye(n)
        R = 0.5 * (seed.R + seed.R.T) + self.psd_jitter * np.eye(p)

 
        params, props = self.model.initialize(
            key_init,
            initial_mean=jnp.zeros(n),
            initial_covariance=jnp.eye(n),
            dynamics_weights=A,
            dynamics_input_weights=B,
            dynamics_covariance=jnp.asarray(Q),
            emission_weights=C,
            emission_input_weights=jnp.zeros((p, m)),   # D ≈ 0 by SSID convention
            emission_covariance=jnp.asarray(R),
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

    def fit(self) -> Tuple[LDSParams, float, np.ndarray]:
        """Run EM from the SSID seed and return (best_params, best_ll, ll_trace)."""
        emissions = jnp.asarray(self.observation)
        inputs_array = jnp.asarray(self.inputs) if self.inputs is not None else None

        best_params, best_ll = None, float("-inf")
        best_llr_trace = np.zeros(self.num_em_iters)

        bar = tqdm(range(self.num_restarts), desc="EM (SSID-init)")
        for restart in bar:
            key = jr.PRNGKey(restart)
            params, props = self._initialise_from_ssid(key, perturb=(restart > 0))

            params, llr = self.model.fit_em(
                params, props,
                emissions=emissions,
                inputs=inputs_array,
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

        return LDSParams.from_dynamax(best_params), best_ll, best_llr_trace